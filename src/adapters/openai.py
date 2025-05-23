import os
import json
import base64
import mimetypes
from typing import Optional, Dict, Any, List, Type
from openai import OpenAI, APIError
from pydantic import BaseModel
from pydantic_ai import patch # Import the patch function
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
        
        # Initialize the OpenAI client
        unpatched_client = OpenAI(api_key=api_key)
        # Patch the client with PydanticAI capabilities
        self.client = patch(unpatched_client)
        print(f"OpenAIAdapter initialized and patched for model: {self.model_name}")

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
        tools_details = input_data.get("tools_details", [])
        file_content_data = input_data.get("file_content")
        pydantic_model_class: Optional[Type[BaseModel]] = input_data.get("pydantic_model_class")

        openai_messages = self._format_messages_for_openai(messages, file_content_data)
        
        cost_info = None # Initialize cost_info

        try:
            if pydantic_model_class:
                # Use the patched client with response_model parameter
                # This call is expected to be handled by the pydantic-ai patch,
                # which will guide the LLM to return JSON and parse it into the model.
                # The patched method should ideally return (PydanticModelInstance, RawCompletionData)
                
                # The test mocks `self.client.chat.completions.create`.
                # If `self.client` is patched, this mocked method will be called by pydantic-ai's machinery.
                # The mock returns `mock_openai_response` which contains the JSON string in `choices[0].message.content`
                # and `usage` data. Pydantic-ai's patch should parse this content.

                # The `response_model` parameter is key here for the patched client.
                response_tuple = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    response_model=pydantic_model_class, 
                    # Tools might be incompatible with response_model in some patch versions,
                    # or require specific handling. The test focuses on response_model.
                    # If tools_details are present, this might need adjustment or error handling.
                )

                # Assuming the patched client returns (model_instance, original_completion_object)
                # This is a common pattern for instructor-like libraries.
                if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                    model_instance, completion = response_tuple
                else:
                    # Fallback if the patch returns only the model instance (less ideal for cost tracking)
                    model_instance = response_tuple 
                    completion = None # No direct access to original completion object

                if completion and hasattr(completion, 'usage') and completion.usage:
                    cost_info = {
                        "tokens_used": completion.usage.total_tokens,
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "cost": 0 
                    }
                    if self.cost_tracker:
                        calculated_cost = self.cost_tracker.calculate_cost(
                            self.model_name, 
                            completion.usage.prompt_tokens, 
                            completion.usage.completion_tokens
                        )
                        if calculated_cost is not None:
                            cost_info["cost"] = calculated_cost
                else:
                    # Fallback if usage info is not available
                    cost_info = {"tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0, "warning": "Usage data not available or patch signature mismatch"}

                return {
                    "content": {"model_instance": model_instance, "text": None, "tool_calls": []},
                    "cost_info": cost_info
                }
            else:
                # Standard processing for text or tool calls (no Pydantic model requested, use unpatched behavior)
                # Note: self.client is already patched. If we need unpatched, we'd have to manage two clients.
                # For now, assume standard calls on patched client behave normally if response_model is not given.
                # Or, the patch might require response_model. This needs testing.
                # If standard calls fail on patched client without response_model, this logic is flawed.
                api_params = {
                    "model": self.model_name,
                    "messages": openai_messages,
                }
                if tools_details:
                    api_params["tools"] = tools_details
                    api_params["tool_choice"] = "auto"

                completion = self.client.chat.completions.create(**api_params)
                response_message = completion.choices[0].message
                
                text_content = response_message.content
                tool_calls = []

                if response_message.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                        } for tc in response_message.tool_calls
                    ]
                    text_content = None

                if completion.usage:
                    cost_info = {
                        "tokens_used": completion.usage.total_tokens,
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "cost": 0
                    }
                    if self.cost_tracker:
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

        except APIError as e:
            print(f"OpenAI API Error: {e}")
            return {
                "content": {"text": f"OpenAI API Error: {e}", "model_instance": None, "tool_calls": []},
                "cost_info": None
            }
        except Exception as e: # Catch other potential errors, e.g., from PydanticAI
            print(f"Error during OpenAIAdapter process: {e}")
            return {
                "content": {"text": f"Adapter processing error: {e}", "model_instance": None, "tool_calls": []},
                "cost_info": None
            }
