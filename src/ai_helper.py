import json
import inspect # For tool schema generation
from typing import Type, TypeVar, Optional, List, Dict, Any, Callable
from pydantic import BaseModel, ValidationError

# from py_models.pd_reader_model import PDReaderModel # File does not exist
from py_models.weather_model import WeatherModel # Example
from src.cost_tracker import CostTracker
from src.adapters.base_adapter import BaseAdapter
from src.adapters.openai import OpenAIAdapter
from src.adapters.google import GoogleAdapter
from src.adapters.anthropic import AnthropicAdapter
from src.adapters.openrouter import OpenRouterAdapter
# from src.tools import calculator, weather, pdf_reader # Keep for later tool implementation

# Define a type variable for Pydantic models
PydanticModelType = TypeVar("PydanticModelType", bound=BaseModel)

class AiHelper:
    def __init__(self, model_identifier: str, cost_tracker: Optional[CostTracker] = None):
        self.model_identifier = model_identifier
        self.cost_tracker = cost_tracker if cost_tracker is not None else CostTracker()
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.adapter: BaseAdapter = self._initialize_adapter()

    def _initialize_adapter(self) -> BaseAdapter:
        parts = self.model_identifier.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Model identifier '{self.model_identifier}' is not in 'provider:model_name' format.")
        
        provider, model_name = parts
        
        if provider == "openai":
            return OpenAIAdapter(model_name=model_name, cost_tracker=self.cost_tracker)
        elif provider == "google":
            return GoogleAdapter(model_name=model_name, cost_tracker=self.cost_tracker)
        elif provider == "anthropic":
            return AnthropicAdapter(model_name=model_name, cost_tracker=self.cost_tracker)
        elif provider == "openrouter":
            return OpenRouterAdapter(model_name=model_name, cost_tracker=self.cost_tracker)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def add_tool(self, name: str, description: str, func: Callable): # Ensure func is callable
        self.available_tools[name] = {"description": description, "func": func}

    def _generate_tool_schema(self, func: Callable, name: str, description: str) -> Dict[str, Any]:
        """Generates a basic JSON schema for a tool's parameters."""
        params = inspect.signature(func).parameters
        properties = {}
        required_params = []
        for param_name, param_obj in params.items():
            # Basic type mapping; extend as needed
            param_type = "string" # Default
            if param_obj.annotation == int:
                param_type = "integer"
            elif param_obj.annotation == float:
                param_type = "number"
            elif param_obj.annotation == bool:
                param_type = "boolean"
            
            properties[param_name] = {"type": param_type}
            # Consider description for params if available from docstring parsing (more complex)

            if param_obj.default == inspect.Parameter.empty:
                required_params.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            }
        }

    def ask(self, 
            prompt: str, 
            tools: Optional[List[str]] = None, 
            pydantic_model: Optional[Type[PydanticModelType]] = None, 
            file_path: Optional[str] = None
           ) -> Any:

        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        # TODO: If file_path is provided, add its content to the first user message
        # This will depend on how adapters handle multimodal inputs.
        # For now, file_path is passed separately in input_data for initial processing.

        max_tool_calls = 5 # To prevent infinite loops
        tool_call_count = 0
        
        while tool_call_count < max_tool_calls:
            input_data: Dict[str, Any] = {"messages": list(messages)} # Pass a copy

            if pydantic_model:
                # Pass the Pydantic model class itself to the adapter
                input_data["pydantic_model_class"] = pydantic_model

            current_tools_details = []
            if tools: # Only pass tool schemas on the first turn or if explicitly re-requested
                for tool_name in tools:
                    if tool_name in self.available_tools:
                        tool_func = self.available_tools[tool_name]["func"]
                        tool_desc = self.available_tools[tool_name]["description"]
                        current_tools_details.append(self._generate_tool_schema(tool_func, tool_name, tool_desc))
            input_data["tools_details"] = current_tools_details
            
            # File handling (simplified for now, might need to be part of messages for some models)
            if file_path and tool_call_count == 0: # Only process file on first turn
                try:
                    with open(file_path, "rb") as f:
                        input_data["file_content"] = {
                            "filename": file_path.split('/')[-1],
                            "content_bytes": f.read()
                        }
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    # Decide how to handle: proceed without file, or raise error?
                    # For now, messages list will not contain file info if read fails.

            adapter_response = self.adapter.process(input_data)
            
            llm_text_response = ""
            tool_calls_data = [] # Expected format: [{"id": "call_abc", "type": "function", "function": {"name": "tool_name", "arguments": '{"arg1": "val1"}'}}]
            
            if adapter_response and isinstance(adapter_response, dict):
                content_part = adapter_response.get("content", {})
                if isinstance(content_part, dict):
                    llm_text_response = content_part.get("text", "")
                    tool_calls_data = content_part.get("tool_calls", [])
                
                cost_info = adapter_response.get("cost_info")
                if cost_info:
                    self._track_cost(cost_info)

            # Add LLM's response (text or tool call request) to messages
            # Adapters should ideally return a message object that can be directly appended.
            # For now, construct it here.
            llm_response_message = {"role": "assistant", "content": llm_text_response}
            if tool_calls_data:
                llm_response_message["tool_calls"] = tool_calls_data
            messages.append(llm_response_message)

            if not tool_calls_data: # No tool calls, LLM provided a final text response
                break 
            
            # If there are tool calls, execute them
            tool_call_count += 1
            tool_results_messages = []
            for tool_call in tool_calls_data:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_id = tool_call.get("id") # Important for some APIs like OpenAI

                if tool_name and tool_name in self.available_tools:
                    try:
                        tool_args = json.loads(tool_args_str)
                        tool_result = self._execute_tool(tool_name, tool_args)
                    except json.JSONDecodeError as e:
                        tool_result = f"Error: Invalid JSON arguments for tool {tool_name}: {e}"
                    except Exception as e: # Catch errors from _execute_tool itself
                        tool_result = str(e)
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found or definition mismatch."
                
                tool_results_messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call_id, 
                    "name": tool_name, 
                    "content": str(tool_result) # Ensure content is string
                })
            
            messages.extend(tool_results_messages)
            # Loop back to call LLM with tool results

            # After loop (either final response or max_tool_calls reached)
        if pydantic_model:
            # Adapter is now expected to return a Pydantic model instance directly
            # in adapter_response["content"]["model_instance"]
            if adapter_response and isinstance(adapter_response, dict):
                content_part = adapter_response.get("content", {})
                if isinstance(content_part, dict):
                    model_instance = content_part.get("model_instance")
                    if model_instance and isinstance(model_instance, pydantic_model):
                        # TODO: Add logic for "how many percent of model fields are filled"
                        # This would involve inspecting model_instance.__fields_set__
                        # and comparing with pydantic_model.model_fields.
                        return model_instance
                    elif model_instance is not None: # It's something, but not the expected model type
                        print(f"Warning: Adapter returned a model_instance of type {type(model_instance)}, expected {pydantic_model}.")
                        # Fallback or error? For now, try to return it if it's a BaseModel at least.
                        if isinstance(model_instance, BaseModel):
                            return model_instance 
                        return pydantic_model.model_construct() # Return empty/default model
                
            # If model_instance was not found or not of the correct type in the response
            print(f"Warning: Pydantic model {pydantic_model.__name__} requested, but not found or invalid in adapter response.")
            return pydantic_model.model_construct() # Return empty/default model
        else:
            # No Pydantic model specified, return the raw text response from the last LLM interaction
            # llm_text_response should already be set from the loop
            return llm_text_response

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        if tool_name in self.available_tools:
            tool_func = self.available_tools[tool_name]["func"]
            try:
                # Consider security implications if tool_args are directly from LLM
                return tool_func(**tool_args)
            except Exception as e:
                return f"Error executing tool {tool_name}: {e}"
        return f"Tool {tool_name} not found."

    def _track_cost(self, cost_details: Dict[str, Any]):
        if self.cost_tracker and cost_details: # Ensure cost_tracker exists and details are provided
            self.cost_tracker.track_cost(cost_details)
