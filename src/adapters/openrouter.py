import os
import json
from src.adapters.base_adapter import BaseAdapter
from openai import OpenAI # Import the OpenAI client for OpenRouter
from pydantic import BaseModel # Import BaseModel to check for Pydantic models

class OpenRouterAdapter(BaseAdapter):
    def __init__(self):
        # Initialize the OpenAI client configured for OpenRouter
        # TODO: Handle missing API key more gracefully
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        # TODO: Allow model to be configured

    def process(self, input_data):
        print(f"Processing data with OpenRouterAdapter: {input_data}")

        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Prompt is missing in input_data"}

        messages = [{"role": "user", "content": prompt}]
        tools_for_api = None
        tool_choice_for_api = "none" # Default to no tool
        output_model_name = input_data.get("output_model")

        # Handle tools if provided in input_data
        tools_info = input_data.get("tools")
        if tools_info:
            # Format tools for the OpenAI API (used by OpenRouter)
            tools_for_api = []
            for tool in tools_info:
                # Assuming tool_info has 'name' and 'description'
                # For function calling, we need a 'function' object with 'name', 'description', and 'parameters'
                # This requires a way to get the schema of the tool's arguments.
                # For now, we'll create a basic function definition.
                # TODO: Get actual parameter schema for tools
                tools_for_api.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": { # Placeholder parameters
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                })
            tool_choice_for_api = "auto" # Allow the model to choose a tool automatically

        # Handle Pydantic model if provided in input_data
        if output_model_name:
             # If a Pydantic model is expected, try to guide the model to produce JSON
             # This can be done using function calling or response_format in newer models
             # For now, let's add a message asking for JSON output and set response_format if model supports it
             # TODO: Use function calling with Pydantic schema or response_format="json_object"
             messages[0]["content"] += f"\n\nPlease provide the response as a JSON object conforming to the schema for {output_model_name}."
             # Note: response_format="json_object" is only supported by certain models (e.g., gpt-4-1106-preview, gpt-3.5-turbo-1106)
             # We would need to check the model capabilities or use a model that supports it.


        # TODO: Handle file content in input_data (e.g., by adding to messages or using vision models)

        try:
            # Make the actual API call
            # Include tools and tool_choice if available
            response = self.client.chat.completions.create(
                model="openrouter/auto", # Example model, should be configurable
                messages=messages,
                tools=tools_for_api, # Pass tools to the API
                tool_choice=tool_choice_for_api # Set tool choice
                # TODO: Add response_format="json_object" for structured output if model supports it
            )

            # Extract relevant information from the response
            # Assuming the response structure from the OpenAI library (used by OpenRouter)
            message = response.choices[0].message # Assuming at least one choice

            model_output = None
            tool_calls = None

            if message.tool_calls:
                # If the model requested tool calls
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "args": json.loads(tool_call.function.arguments) # Assuming arguments are JSON string
                    })
                # TODO: Execute tools and potentially call the API again with tool results

            if message.content:
                # If the model returned content
                model_output = message.content

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
