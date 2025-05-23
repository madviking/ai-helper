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
        
        if output_model:
            response = self._validate_model(response, output_model)
        
        if self.cost_tracker:
            self._track_cost({"request": prompt, "response": response})
        
        return response

    def _execute_tool(self, tool_name, tool_args):
        """Execute a specific tool with given arguments."""
        for tool in self.tools:
            if tool["name"] == tool_name:
                return tool["function"](tool_args)
        return None

    def _track_cost(self, cost_details):
        """Track the cost of the request if a cost tracker is provided."""
        if self.cost_tracker:
            self.cost_tracker.track_cost(cost_details)

    def _validate_model(self, response, output_model):
        """Validate the response against the specified output model and calculate field fill percentage."""
        # Placeholder for actual validation logic
        # For now, just return the response as is
        return response
