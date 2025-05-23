import os
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel, ValidationError
import json

from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter
from src.adapters.openai import OpenAIAdapter
from src.adapters.anthropic import AnthropicAdapter
from src.adapters.google import GoogleAdapter
from src.adapters.openrouter import OpenRouterAdapter
from src.tools import calculator, weather, pdf_reader


class AIHelper:
    def __init__(self, model_identifier: str, cost_tracker: CostTracker = None):
        self.model_identifier = model_identifier
        self.cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self.available_tools = {}
        
        # Parse model identifier
        self._parse_model_identifier()
        self._initialize_adapter()

    def _parse_model_identifier(self):
        """Parse the model identifier to extract provider and model name."""
        if ':' not in self.model_identifier:
            raise ValueError(f"Invalid model identifier format: {self.model_identifier}. Expected format: 'provider:model'")
        
        parts = self.model_identifier.split(':', 1)
        self.provider = parts[0]
        self.model = parts[1]
        
        valid_providers = ['openai', 'anthropic', 'google', 'openrouter']
        if self.provider not in valid_providers:
            raise ValueError(f"Unknown provider: {self.provider}. Valid providers are: {', '.join(valid_providers)}")

    def _initialize_adapter(self):
        """Initialize the appropriate adapter based on the provider."""
        if self.provider == 'openai':
            self.adapter = OpenAIAdapter()
        elif self.provider == 'anthropic':
            self.adapter = AnthropicAdapter()
        elif self.provider == 'google':
            self.adapter = GoogleAdapter()
        elif self.provider == 'openrouter':
            self.adapter = OpenRouterAdapter()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def add_tool(self, name: str, description: str, func=None):
        """Add a tool to the available tools."""
        # If func is not provided, try to get it from the tools module
        if func is None:
            if name == 'calculator':
                func = calculator
            elif name == 'weather':
                func = weather
            elif name == 'pdf_reader':
                func = pdf_reader
            else:
                raise ValueError(f"Unknown tool name: {name}")
        
        self.available_tools[name] = {"description": description, "func": func}

    def ask(self, prompt: str, tools: List[str] = None, model: Type[BaseModel] = None, file: str = None):
        """
        Process a request using the configured adapter.
        
        Args:
            prompt: The prompt to send to the LLM
            tools: List of tool names to make available
            model: Pydantic model class to validate the response
            file: Path to a file to include in the request
            
        Returns:
            Pydantic model instance if model is provided, otherwise raw response
        """
        # Check for file support
        if file and self.provider == 'anthropic':
            raise ValueError(f"Provider '{self.provider}' does not support file input")
        
        # Prepare tools dictionary for the adapter
        tools_dict = None
        if tools:
            tools_dict = {}
            for tool_name in tools:
                if tool_name not in self.available_tools:
                    raise ValueError(f"Tool '{tool_name}' not found in available tools")
                tools_dict[tool_name] = self.available_tools[tool_name]['func']
        
        # Prepare input data for adapter
        input_data = {
            'prompt': prompt,
            'model': self.model,
            'pydantic_model': model,
            'tools': tools_dict,
            'file': file
        }
        
        # Process through adapter
        result = self.adapter.process(input_data)
        
        # Track costs
        self.cost_tracker.add_cost(
            input_tokens=result['input_tokens'],
            output_tokens=result['output_tokens'],
            pydantic_model=model.__name__ if model else None,
            llm_model=result.get('model_used', self.model)
        )
        
        # Return the response
        return result['response']

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]):
        """Execute a specific tool from available tools."""
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' not found in available tools")
        
        tool_func = self.available_tools[tool_name]['func']
        return tool_func(**tool_args)

    def _track_cost(self, cost_details: Dict[str, Any]):
        """Track cost using the cost tracker."""
        self.cost_tracker.track_cost(cost_details)
