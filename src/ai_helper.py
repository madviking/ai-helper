# Temporarily commented out due to module not found error
# from py_models.pd_reader_model import PDReaderModel
# from py_models.weather_model import WeatherModel
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter
from src.tools import calculator, weather, pdf_reader # Import specific tools

class AiHelper:
    def __init__(self, model_identifier: str, cost_tracker: CostTracker = None, adapter: BaseAdapter = None):
        self.model_identifier = model_identifier
        self.cost_tracker = cost_tracker
        self.tools = []
        self.adapter = adapter
        if not adapter:
            self._initialize_adapter()

    def _initialize_adapter(self):
        # Logic to select and initialize the appropriate adapter based on self.model_identifier
        from src.adapters.anthropic import AnthropicAdapter
        from src.adapters.google import GoogleAdapter
        from src.adapters.openai import OpenAIAdapter
        from src.adapters.openrouter import OpenRouterAdapter
        
        if "anthropic" in self.model_identifier.lower():
            self.adapter = AnthropicAdapter(self.model_identifier)
        elif "google" in self.model_identifier.lower():
            self.adapter = GoogleAdapter(self.model_identifier)
        elif "openai" in self.model_identifier.lower():
            self.adapter = OpenAIAdapter(self.model_identifier)
        else:
            self.adapter = OpenRouterAdapter()

    def add_tool(self, tool: dict):
        """Add a tool to the tools list."""
        if "name" in tool and "function" in tool:
            self.tools.append(tool)

    def ask(self, prompt: str, tools: list = None, output_model=None, file_path: str = None):
        """Process a request to the AI model with optional tools, output model, and file input."""
        input_data = {"prompt": prompt}
        if file_path:
            input_data["file_path"] = file_path
        if tools:
            input_data["tools"] = tools
        
        response = self.adapter.process(input_data)
        
        # Extract token information for cost tracking if available
        input_tokens = response.get("input_tokens", 0)
        output_tokens = response.get("output_tokens", 0)
        
        # Check if the response indicates a tool call (assuming tool call info is in response)
        content = response.get("content", "")
        if tools and "tool_call" in content.lower():
            import json
            try:
                tool_data = json.loads(content)
                if "tool_name" in tool_data and "tool_args" in tool_data:
                    tool_result = self._execute_tool(tool_data["tool_name"], tool_data["tool_args"])
                    if output_model:
                        response = self._validate_model(tool_result, output_model)
                    else:
                        response = {"content": tool_result, "input_tokens": input_tokens, "output_tokens": output_tokens}
            except json.JSONDecodeError:
                pass
        elif output_model:
            # If an output model is specified and no tool call, validate the response content
            content = response.get("content", "")
            response = self._validate_model(content, output_model)
        
        if self.cost_tracker:
            self._track_cost({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "model": self.model_identifier
            })
        
        return response

    def _execute_tool(self, tool_name, tool_args):
        """Execute a specific tool with given arguments."""
        for tool in self.tools:
            if tool["name"] == tool_name:
                # If tool_args is a dictionary, try to extract a single input if applicable
                if isinstance(tool_args, dict) and len(tool_args) == 1 and "input" in tool_args:
                    return tool["function"](tool_args["input"])
                return tool["function"](tool_args)
        return None

    def _track_cost(self, cost_details):
        """Track the cost of the request if a cost tracker is provided."""
        if self.cost_tracker:
            self.cost_tracker.track_cost(cost_details)

    def _validate_model(self, response, output_model):
        """Validate the response against the specified output model and calculate field fill percentage."""
        import json
        from pydantic import ValidationError
        
        try:
            # Attempt to parse the response as JSON
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            # Validate the data against the output model
            model_instance = output_model(**data)
            
            # Calculate the percentage of fields filled
            total_fields = len(output_model.model_fields)
            filled_fields = 0
            for field_name, field_info in output_model.model_fields.items():
                value = getattr(model_instance, field_name)
                if value != field_info.default:
                    filled_fields += 1
            completion_percentage = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
            
            return {
                "model": model_instance,
                "completion_percentage": completion_percentage
            }
        except (json.JSONDecodeError, ValidationError):
            # If parsing or validation fails, attempt to create a model with default values
            # and fill in any valid fields from the response
            model_instance = output_model()
            if isinstance(response, str):
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    data = {}
            else:
                data = response if isinstance(response, dict) else {}
                
            filled_fields = 0
            for field_name, field_info in output_model.model_fields.items():
                if field_name in data:
                    try:
                        # Validate the field value against the field's type before setting
                        temp_model = output_model(**{field_name: data[field_name]})
                        setattr(model_instance, field_name, data[field_name])
                        if getattr(model_instance, field_name) != field_info.default:
                            filled_fields += 1
                    except (TypeError, ValueError, ValidationError):
                        # Discard invalid field data by not setting it
                        pass
            
            total_fields = len(output_model.model_fields)
            completion_percentage = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
            
            return {
                "model": model_instance,
                "completion_percentage": completion_percentage
            }
