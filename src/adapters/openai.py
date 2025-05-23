import os
import json
import base64
import mimetypes
from typing import Optional, Dict, Any, List
from openai import OpenAI, APIError # Import OpenAI and APIError
from src.adapters.base_adapter import BaseAdapter
from src.cost_tracker import CostTracker

# Helper to guess MIME type
def get_mime_type(filename: str) -> Optional[str]:
    return mimetypes.guess_type(filename)[0]

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model_name: str, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, cost_tracker)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        print(f"OpenAIAdapter initialized for model: {self.model_name}")

    def _format_messages_for_openai(self, messages: List[Dict[str, Any]], file_content_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") # For assistant messages
            tool_call_id = msg.get("tool_call_id") # For tool messages
            name = msg.get("name") # For tool messages (tool name)

            openai_msg = {"role": role}

            # Handle file input for the first user message if it's an image
            # OpenAI expects multimodal content as a list under the 'content' key for user messages
            if role == "user" and i == 0 and file_content_data: # Assuming file is for the initial user prompt
                filename = file_content_data.get("filename")
                content_bytes = file_content_data.get("content_bytes")
                mime_type = get_mime_type(filename) if filename else None

                # Basic image handling (extend for more types if needed)
                if mime_type and mime_type.startswith("image/") and content_bytes:
                    base64_image = base64.b64encode(content_bytes).decode('utf-8')
                    image_url = f"data:{mime_type};base64,{base64_image}"
                    
                    # Construct multimodal content
                    multi_content = [{"type": "text", "text": content if isinstance(content, str) else ""}]
                    multi_content.append({"type": "image_url", "image_url": {"url": image_url}})
                    openai_msg["content"] = multi_content
                elif isinstance(content, str): # Fallback to text content if not a recognized image
                    openai_msg["content"] = content
                else: # If content is not string and not a handled image, ensure it's at least an empty string
                    openai_msg["content"] = ""

            elif isinstance(content, str):
                openai_msg["content"] = content
            
            if tool_calls: # Assistant message requesting tool calls
                openai_msg["tool_calls"] = tool_calls
            
            if role == "tool" and tool_call_id: # Tool execution result message
                openai_msg["tool_call_id"] = tool_call_id
                openai_msg["name"] = name # Function name
                # Content for tool role is the result of the tool execution
                openai_msg["content"] = str(content) if content is not None else ""


            # Ensure content is not None if not handled above (e.g. assistant message with only tool_calls)
            if "content" not in openai_msg and role == "assistant" and not tool_calls:
                 openai_msg["content"] = "" # OpenAI requires content for assistant messages unless it's only tool_calls
            elif "content" not in openai_msg and role != "tool" and role != "assistant": # e.g. user message without file
                 openai_msg["content"] = str(content) if content is not None else ""


            # Filter out messages that might be empty in a way OpenAI dislikes
            if role == "assistant" and not openai_msg.get("content") and not openai_msg.get("tool_calls"):
                continue # Skip assistant messages that are truly empty
            
            formatted_messages.append(openai_msg)
            
        return formatted_messages


    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = input_data.get("messages", [])
        tools_details = input_data.get("tools_details", []) # Schema from AiHelper
        file_content_data = input_data.get("file_content") # {"filename": ..., "content_bytes": ...}
        # pydantic_model_json_schema = input_data.get("pydantic_model_json_schema") # For instructing JSON mode

        openai_messages = self._format_messages_for_openai(messages, file_content_data)
        
        api_params = {
            "model": self.model_name,
            "messages": openai_messages,
        }

        if tools_details:
            api_params["tools"] = tools_details
            api_params["tool_choice"] = "auto" # Or specific tool if needed

        # TODO: Add support for JSON mode if pydantic_model_json_schema is present
        # if pydantic_model_json_schema and self.model_name supports JSON mode (e.g. gpt-3.5-turbo-1106 onwards):
        #    api_params["response_format"] = {"type": "json_object"}

        try:
            completion = self.client.chat.completions.create(**api_params)
        except APIError as e:
            print(f"OpenAI API Error: {e}")
            # Return a structured error or raise it
            return {
                "content": {"text": f"OpenAI API Error: {e}", "tool_calls": []},
                "cost_info": None
            }

        response_message = completion.choices[0].message
        
        text_content = response_message.content
        tool_calls = []

        if response_message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type, # Should be 'function'
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in response_message.tool_calls
            ]
            text_content = None # Usually no text content when tool_calls are present

        cost_info = None
        if completion.usage:
            # This is a simplified cost calculation.
            # Actual costs depend on model, input/output tokens.
            # For now, just passing token counts. AiHelper's CostTracker would need model-specific rates.
            cost_info = {
                "tokens_used": completion.usage.total_tokens,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "cost": 0 # Placeholder: CostTracker should calculate this
            }
            if self.cost_tracker: # Let CostTracker calculate if it can
                calculated_cost = self.cost_tracker.calculate_cost(
                    self.model_name, 
                    completion.usage.prompt_tokens, 
                    completion.usage.completion_tokens
                )
                if calculated_cost is not None:
                    cost_info["cost"] = calculated_cost


        return {
            "content": {"text": text_content, "tool_calls": tool_calls},
            "cost_info": cost_info
        }
