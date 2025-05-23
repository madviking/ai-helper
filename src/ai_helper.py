import json
from py_models.pd_reader_model import PDReaderModel
from py_models.weather_model import WeatherModel
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter
from src.adapters.anthropic import AnthropicAdapter
from src.adapters.google import GoogleAdapter
from src.adapters.openai import OpenAIAdapter
from src.adapters.openrouter import OpenRouterAdapter
from src.tools import calculator, weather, pdf_reader # Import specific tools
from pydantic import BaseModel, ValidationError

class AiHelper:
    def __init__(self, model_identifier: str, cost_tracker: CostTracker = None):
        self.model_identifier = model_identifier
        self.cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self.available_tools = {}
        self.adapter: BaseAdapter = None
        self._initialize_adapter()

    def _initialize_adapter(self):
        # Logic to select and initialize the appropriate adapter based on self.model_identifier
        if self.model_identifier == "anthropic":
            self.adapter = AnthropicAdapter()
        elif self.model_identifier == "google":
            self.adapter = GoogleAdapter()
        elif self.model_identifier == "openai":
            self.adapter = OpenAIAdapter()
        elif self.model_identifier == "openrouter":
            self.adapter = OpenRouterAdapter()
        else:
            raise ValueError(f"Unknown model identifier: {self.model_identifier}")

    def add_tool(self, name: str, description: str, func):
        self.available_tools[name] = {"description": description, "func": func}

    def ask(self, prompt: str, tools: list = None, pydantic_model=None, file: str = None):
        input_data = {"prompt": prompt}

        if file:
            try:
                with open(file, 'r') as f:
                    file_content = f.read()
                input_data["file_content"] = file_content
            except FileNotFoundError:
                return f"Error: File not found at {file}"
            except Exception as e:
                return f"Error reading file {file}: {e}"

        if tools:
            # Include tool information in the input data for the adapter
            input_data["tools"] = [{"name": name, "description": self.available_tools[name]['description']} for name in tools if name in self.available_tools]

        if pydantic_model:
            # Include information about the expected output model
            input_data["output_model"] = pydantic_model.__name__ # Or a schema representation

        # Use the adapter to interact with the LLM
        # The adapter's process method should handle the structure of input_data
        adapter_response = self.adapter.process(input_data)

        # Assuming adapter_response is a dictionary that might contain tool calls or model data
        # The exact format will depend on the adapter implementation and LLM output
        # For now, let's assume a structure like:
        # {"tool_calls": [{"name": "tool_name", "args": {...}}], "model_output": {...}, "cost_details": {...}}

        tool_calls = adapter_response.get("tool_calls")
        model_output_data = adapter_response.get("model_output")
        cost_details = adapter_response.get("cost_details")

        # Track cost if cost details are available
        if cost_details:
            self._track_cost(cost_details)
        else:
             # Placeholder cost tracking if adapter doesn't return details
             self._track_cost({"cost": 0.0, "output_tokens": 0, "model": self.model_identifier})


        # Handle tool calls if present
        if tool_calls and tools: # Only execute tools if they were requested
            tool_results = {}
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                if tool_name and tool_name in tools: # Check if the tool is in the allowed list
                    result = self._execute_tool(tool_name, tool_args)
                    tool_results[tool_name] = result
                else:
                    tool_results[tool_name] = f"Error: Tool '{tool_name}' not requested or not found."
            # TODO: Potentially send tool_results back to the LLM via the adapter for a follow-up response

            # For now, return the tool results
            return {"tool_results": tool_results}


        # Handle Pydantic model output if requested and data is available
        if pydantic_model and model_output_data:
            try:
                # Attempt to parse the model_output_data into the Pydantic model
                # Assuming model_output_data is a dictionary
                response_data = model_output_data

                # Validate and load the model
                # TODO: Revisit the requirement to "discard invalid fields" if model_validate fails.
                # Current approach raises ValidationError for any invalid/missing fields.
                total_fields = len(pydantic_model.model_fields) # Use model_fields
                filled_fields = 0

                try:
                    model_instance = pydantic_model.model_validate(response_data) # Use model_validate
                    # If validation succeeds, count how many fields from the response data were used
                    # This is a simplified way to calculate filled percentage based on successful validation
                    # A more accurate way would involve comparing response_data keys to model_fields keys
                    # after successful validation. For now, assume all fields in response_data that are
                    # also in model_fields contribute to filled_fields if validation passes.
                    filled_fields = len([field for field in response_data if field in pydantic_model.model_fields])


                    filled_percentage = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
                    return {"model": model_instance, "filled_percentage": filled_percentage}

                except ValidationError as e:
                    return f"Pydantic model validation error: {e}"

            except Exception as e:
                return f"Error processing Pydantic model output: {e}"

        # If no tool calls or Pydantic model output, return the raw adapter response or model_output_data
        return adapter_response.get("model_output", adapter_response) # Prefer model_output if available


    def _execute_tool(self, tool_name, tool_args):
        # Logic to execute a specific tool from self.available_tools
        if tool_name in self.available_tools:
            tool_func = self.available_tools[tool_name]["func"]
            try:
                # Assuming tool_args is a dictionary that can be passed as keyword arguments
                return tool_func(**tool_args)
            except Exception as e:
                return f"Error executing tool '{tool_name}': {e}"
        else:
            return f"Error: Tool '{tool_name}' not found."


    def _track_cost(self, cost_details):
        # Ensure cost_details has required keys before passing to cost_tracker
        if all(key in cost_details for key in ["cost", "output_tokens", "model"]):
             self.cost_tracker.track_cost(cost_details["cost"], cost_details["output_tokens"], cost_details["model"])
        else:
             print(f"Warning: Invalid cost details provided: {cost_details}")
