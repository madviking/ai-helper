import os
import json
from src.adapters.base_adapter import BaseAdapter
from openai import OpenAI # Import the OpenAI client for OpenRouter
from pydantic import BaseModel, ValidationError # Import BaseModel and ValidationError
from typing import get_origin, get_args # Import typing helpers for schema

class OpenRouterAdapter(BaseAdapter):
    def __init__(self):
        # Initialize the OpenAI client configured for OpenRouter
        # TODO: Handle missing API key more gracefully
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_KEY"), # Use the correct key name
        )
        # TODO: Allow model to be configured

    def process(self, input_data):
        print(f"Processing data with OpenRouterAdapter: {input_data}")

        prompt = input_data.get("prompt")
        file_content = input_data.get("file_content")
        tools_info = input_data.get("tools")
        output_model_class = input_data.get("output_model") # This will be the Pydantic model class

        if not prompt and not file_content:
            return {"error": "Prompt is missing in input_data and no file content was provided"}

        messages = []
        if prompt:
             messages.append({"role": "user", "content": prompt})
        if file_content:
             # TODO: Handle file content appropriately for the model (e.g., vision models)
             # For now, just append to the prompt
             if prompt:
                 messages[0]["content"] += f"\n\nFile Content:\n{file_content}"
             else:
                 messages.append({"role": "user", "content": f"File Content:\n{file_content}"})


        tools_for_api = None
        tool_choice_for_api = "none" # Default to no tool

        # Handle tools if provided in input_data
        if tools_info: # tools_info is now a list of dictionaries from AiHelper
            # Format tools for the OpenAI API (used by OpenRouter)
            tools_for_api = []
            for tool in tools_info: # Iterate through the list of tool dictionaries
                tool_name = tool.get("name")
                tool_description = tool.get("description", f"Tool to perform {tool_name} operation") # Use provided description or placeholder

                # For function calling, we need a 'function' object with 'name', 'description', and 'parameters'
                # This requires a way to get the schema of the tool's arguments.
                # TODO: Get actual parameter schema for tools
                # For now, we'll create a basic function definition.
                tools_for_api.append({
                    "type": "function",
                    "function": {
                        "name": tool_name, # Use the tool name from the dictionary
                        "description": tool_description, # Use the description from the dictionary
                        "parameters": { # Placeholder parameters
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                })
            tool_choice_for_api = "auto" # Allow the model to choose a tool automatically

        # Handle Pydantic model if provided in input_data
        response_format_for_api = {"type": "text"} # Default response format
        if output_model_class and issubclass(output_model_class, BaseModel):
             # If a Pydantic model is expected, use function calling with the model's schema
             model_schema = output_model_class.model_json_schema() # Get Pydantic V2 schema
             function_name = f"extract_{output_model_class.__name__.lower()}" # Example function name
             function_description = f"Extract information into a {output_model_class.__name__} object."

             # Format the Pydantic schema as a function definition
             model_tool_for_api = {
                 "type": "function",
                 "function": {
                     "name": function_name,
                     "description": function_description,
                     "parameters": model_schema
                 }
             }

             # Add the model tool to tools_for_api and set tool_choice to force this function call
             if tools_for_api is None:
                 tools_for_api = []
             tools_for_api.append(model_tool_for_api)
             tool_choice_for_api = {"type": "function", "function": {"name": function_name}} # Force the model to call this function


        # TODO: Handle file content in input_data (e.g., by adding to messages or using vision models)

        try:
            # Make the actual API call
            # Include tools, tool_choice, and response_format if available
            create_params = {
                "model": "openrouter/auto", # Example model, should be configurable
                "messages": messages,
                # response_format is not needed when using function calling for structured output
            }
            if tools_for_api: # Only include tools and tool_choice if tools are specified
                 create_params["tools"] = tools_for_api
                 create_params["tool_choice"] = tool_choice_for_api # Use the determined tool_choice

            response = self.client.chat.completions.create(**create_params)


            # Extract relevant information from the response
            # Assuming the response structure from the OpenAI library (used by OpenRouter)
            message = response.choices[0].message # Assuming at least one choice

            model_output = None
            tool_calls = None
            pydantic_model_instance = None # To store the parsed Pydantic model

            if message.tool_calls:
                # If the model requested tool calls
                tool_calls = []
                for tool_call in message.tool_calls:
                    try:
                        tool_args = json.loads(tool_call.function.arguments) # Assuming arguments are JSON string
                        tool_calls.append({
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "args": tool_args
                        })

                        # If this tool call is for the Pydantic model extraction function
                        if output_model_class and tool_call.function.name == f"extract_{output_model_class.__name__.lower()}":
                             # Attempt to validate and load the Pydantic model from the tool arguments
                             try:
                                 pydantic_model_instance = output_model_class.model_validate(tool_args) # Use model_validate
                                 # TODO: Calculate filled percentage based on tool_args and model_fields
                                 total_fields = len(output_model_class.model_fields)
                                 filled_fields = len([field for field in tool_args if field in output_model_class.model_fields])
                                 filled_percentage = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
                                 # Return the parsed model instance and percentage directly
                                 return {"model": pydantic_model_instance, "filled_percentage": filled_percentage}
                             except ValidationError as e:
                                 return f"Pydantic model validation error from tool call: {e}"
                             except Exception as e:
                                 return f"Error processing Pydantic model tool call: {e}"


                    except json.JSONDecodeError:
                        tool_calls.append({
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "args": None, # Indicate invalid JSON
                            "error": "Invalid JSON arguments for tool call"
                        })
                    except Exception as e:
                         tool_calls.append({
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "args": None,
                            "error": f"Error processing tool call arguments: {e}"
                        })

                # TODO: Execute tools and potentially call the API again with tool results

            if message.content:
                # If the model returned content (this might happen even with tool calls)
                model_output = message.content

            # Fallback for Pydantic model if no tool call was made for it but model output is present
            if output_model_class and model_output and not tool_calls:
                 try:
                     # Attempt to parse the model_output string as JSON
                     response_data = json.loads(model_output)
                     # Attempt to validate and load the Pydantic model from the parsed JSON
                     model_instance = output_model_class.model_validate(response_data) # Use model_validate
                     # TODO: Calculate filled percentage based on response_data and model_fields
                     total_fields = len(output_model_class.model_fields)
                     filled_fields = len([field for field in response_data if field in output_model_class.model_fields])
                     filled_percentage = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
                     # Return the parsed model instance and percentage directly
                     return {"model": model_instance, "filled_percentage": filled_percentage}
                 except (json.JSONDecodeError, ValidationError) as e:
                     # If parsing or validation fails, return the original model output and a warning
                     print(f"Warning: Could not parse model output as Pydantic model: {e}")
                     # Continue to return the raw model_output below
                 except Exception as e:
                      print(f"Warning: Error processing model output for Pydantic model: {e}")
                      # Continue to return the raw model_output below


            usage = getattr(response, 'usage', None)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            total_tokens = getattr(usage, 'total_tokens', 0)

            # TODO: Calculate actual cost based on tokens and model
            # Cost calculation depends on the model and token usage (input + output)
            # For now, keep placeholder cost
            cost_details = {
                "cost": 0.0, # Placeholder cost
                "output_tokens": output_tokens,
                "input_tokens": input_tokens,
                "total_tokens": total_tokens,
                "model": response.model # Use the model name from the response
            }

            # Return a dictionary including model_output, tool_calls, and cost_details
            result = {"cost_details": cost_details}
            if model_output is not None:
                result["model_output"] = model_output
            if tool_calls is not None:
                result["tool_calls"] = tool_calls

            return result


        except Exception as e:
            return {"error": f"Error during OpenRouter API call: {e}"}
