import os
import json
from src.adapters.base_adapter import BaseAdapter
from anthropic import Anthropic # Import the actual Anthropic client
from pydantic import BaseModel # Import BaseModel to check for Pydantic models
from typing import get_origin, get_args # Import typing helpers for schema

class AnthropicAdapter(BaseAdapter):
    def __init__(self):
        # Initialize the Anthropic client with the API key from environment variables
        # TODO: Handle missing API key more gracefully
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        # TODO: Allow model to be configured

    def process(self, input_data):
        print(f"Processing data with AnthropicAdapter: {input_data}")

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
        # Anthropic uses 'tool_choice' parameter similar to OpenAI

        # Handle tools if provided in input_data
        if tools_info:
            # Format tools for the Anthropic API (tools)
            tools_for_api = []
            for tool_name in tools_info: # tools_info is now a list of tool names
                # Assuming tool_info has 'name' and 'description'
                # For tools, we need a 'tool_definition' object with 'name', 'description', and 'input_schema'
                # This requires a way to get the schema of the tool's arguments.
                # TODO: Get actual parameter schema for tools
                # For now, we'll create a basic tool definition.
                tools_for_api.append({
                    "name": tool_name, # Use the tool name directly
                    "description": f"Tool to perform {tool_name} operation", # Placeholder description
                    "input_schema": { # Placeholder input schema
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                })
            # TODO: Set tool_choice based on input_data (e.g., "auto", {"type": "tool", "name": "my_tool"})


        # Handle Pydantic model if provided in input_data
        # Anthropic doesn't have a direct response_format="json_object" like OpenAI
        # We can try to guide the model to produce JSON using the prompt or a system message
        if output_model_class and issubclass(output_model_class, BaseModel):
             # Add a message asking for JSON output
             messages[0]["content"] += f"\n\nPlease provide the response as a JSON object conforming to the schema for {output_model_class.__name__}."
             # TODO: Potentially include the Pydantic schema in the prompt or a system message


        try:
            # Make the actual API call
            # Include tools if available
            response = self.client.messages.create(
                model="claude-3-opus-20240229", # Example model, should be configurable
                max_tokens=1024, # Example max_tokens, should be configurable
                messages=messages,
                tools=tools_for_api # Pass tools to the API
                # TODO: Include tool_choice if needed
            )

            # Extract relevant information from the response
            # Assuming the response structure from the Anthropic library
            model_output = None
            tool_calls = None

            if response.content:
                # Check for tool_use blocks in the content
                tool_calls = []
                model_output_parts = []
                for block in response.content:
                    if block.type == 'tool_use':
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "args": block.input # Anthropic uses 'input' for tool arguments
                        })
                    elif block.type == 'text':
                        model_output_parts.append(block.text)

                if model_output_parts:
                    model_output = "".join(model_output_parts)

                # TODO: Execute tools and potentially call the API again with tool results


            usage = getattr(response, 'usage', None)
            output_tokens = getattr(usage, 'output_tokens', 0)
            input_tokens = getattr(usage, 'input_tokens', 0) # Anthropic uses input_tokens
            total_tokens = output_tokens + input_tokens # Calculate total tokens

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
            if tool_calls: # Check if tool_calls list is not empty
                result["tool_calls"] = tool_calls

            return result


        except Exception as e:
            return {"error": f"Error during Anthropic API call: {e}"}
