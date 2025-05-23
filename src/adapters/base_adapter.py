from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.cost_tracker import CostTracker # Added for type hinting in __init__

class BaseAdapter(ABC):
    def __init__(self, model_name: str, cost_tracker: Optional[CostTracker] = None):
        self.model_name = model_name
        self.cost_tracker = cost_tracker

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data, which may include messages, tool details, 
        pydantic model schemas, and file content.
        
        Expected input_data keys:
        - "messages": List of message dicts (e.g., [{"role": "user", "content": "..."}])
        - "tools_details": Optional list of tool schema dicts.
        - "pydantic_model_json_schema": Optional JSON schema for the final response.
        - "file_content": Optional dict with "filename" and "content_bytes" if a file is provided.

        Expected return format:
        {
            "content": {
                "text": "LLM's text response or None if only tool calls",
                "tool_calls": [ # List of tool call dicts, empty if no tool calls
                    # Example tool_call:
                    # {
                    #   "id": "call_xyz", 
                    #   "type": "function", 
                    #   "function": {"name": "tool_name", "arguments": '{"arg": "value"}'}
                    # }
                ]
            },
            "cost_info": {"tokens_used": X, "cost": Y} # Optional cost information
        }
        """
        pass
