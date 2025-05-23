import os
import json
import base64
import mimetypes
from typing import Optional, Dict, Any, List, Type
from anthropic import Anthropic, APIError
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.anthropic import AnthropicModel # Corrected import based on file content

from src.adapters.base_adapter import BaseAdapter
from src.cost_tracker import CostTracker

# Helper to guess MIME type
def get_mime_type(filename: str) -> Optional[str]:
    return mimetypes.guess_type(filename)[0]

class AnthropicAdapter(BaseAdapter):
    def __init__(self, model_name: str, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, cost_tracker)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = Anthropic(api_key=api_key)
        print(f"AnthropicAdapter initialized for model: {self.model_name}")

    def _format_messages_for_anthropic(self, messages: List[Dict[str, Any]], file_content_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        anthropic_messages = []
        for i, msg in enumerate(messages):
            role = msg.get("role") # user, assistant (tool results are user messages with tool_result type)
            content = msg.get("content")
            
            # Anthropic expects 'user' or 'assistant' roles.
            # Tool results are sent as a user message containing tool_result content blocks.
            # Tool calls by assistant are part of assistant message content.
            if role == "tool": # Convert AiHelper 'tool' role to Anthropic's format
                anthropic_msg = {"role": "user", "content": []}
                anthropic_msg["content"].append({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"), # Anthropic uses tool_use_id
                    "content": str(content) if content is not None else "",
                    # "is_error": False # Optionally indicate if tool execution failed
                })
                anthropic_messages.append(anthropic_msg)
                continue

            # Regular user/assistant messages
            anthropic_msg_content = []
            
            # Handle file input for the first user message if it's an image
            if role == "user" and i == 0 and file_content_data:
                filename = file_content_data.get("filename")
                content_bytes = file_content_data.get("content_bytes")
                mime_type = get_mime_type(filename) if filename else None

                if mime_type and mime_type.startswith("image/") and content_bytes:
                    base64_image = base64.b64encode(content_bytes).decode('utf-8')
                    anthropic_msg_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_image,
                        },
                    })
            
            # Add text content if any
            if isinstance(content, str) and content:
                anthropic_msg_content.append({"type": "text", "text": content})
            
            # Handle tool calls requested by assistant
            if role == "assistant" and msg.get("tool_calls"):
                for tool_call in msg.get("tool_calls", []): # These are from AiHelper's generic format
                    # Convert to Anthropic's tool_use format if needed, or expect adapter to handle
                    # For now, assuming AiHelper's tool_calls are compatible or will be transformed by adapter
                    # Anthropic's response has 'tool_use' blocks.
                    # If AiHelper sends tool_calls in assistant message, it's for context.
                    # The actual request from LLM is parsed from response.
                    # This part of message formatting might need adjustment based on how AiHelper structures
                    # assistant messages that *contain* tool call requests.
                    # For now, if text is also present, it's added. If only tool_calls, content might be just those.
                    # Anthropic's API expects assistant messages that request tool use to have content blocks of type 'tool_use'.
                    # This formatting step assumes the input `msg["tool_calls"]` is already in Anthropic's desired format
                    # if it's an assistant message *representing* a tool request.
                    # This is tricky because AiHelper's `messages` list is generic.
                    # Let's assume for now that if `msg["tool_calls"]` is present on an assistant message,
                    # it's for context, and the adapter will parse new tool_use from response.
                    # If an assistant message *is* a tool request, its content should be tool_use blocks.
                    # The current AiHelper structure adds tool_calls as a separate key, not directly in content.
                    # This needs careful alignment.
                    pass # Deferring complex assistant tool_call re-formatting for now.

            if anthropic_msg_content: # Only add if there's content
                anthropic_messages.append({"role": role, "content": anthropic_msg_content})
            elif role == "assistant" and not anthropic_msg_content and msg.get("tool_calls"):
                # If an assistant message from history *was* a tool call request, reconstruct it
                # This assumes msg["tool_calls"] is in a format that can be mapped to Anthropic's tool_use blocks
                # This is a simplification.
                reconstructed_tool_uses = []
                for tc in msg.get("tool_calls"):
                    reconstructed_tool_uses.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": tc.get("function", {}).get("name"),
                        "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
                    })
                if reconstructed_tool_uses:
                     anthropic_messages.append({"role": "assistant", "content": reconstructed_tool_uses})


        return anthropic_messages

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = input_data.get("messages", [])
        tools_details_aihelper = input_data.get("tools_details", [])
        file_content_data = input_data.get("file_content")
        pydantic_model_class: Optional[Type[BaseModel]] = input_data.get("pydantic_model_class")
        system_prompt = None # Anthropic uses a dedicated 'system' parameter

        # Note: _format_messages_for_anthropic is for the direct API call.
        # pydantic-ai.Agent might expect a simpler list of strings/BinaryContent.
        
        cost_info = None

        try:
            if pydantic_model_class:
                agent_input_list = []
                # Handle file content for Agent
                if file_content_data:
                    content_bytes = file_content_data.get("content_bytes")
                    filename = file_content_data.get("filename")
                    mime_type = get_mime_type(filename) if filename else "application/octet-stream"
                    if content_bytes:
                        if mime_type and mime_type.startswith("image/"):
                            agent_input_list.append(BinaryContent(data=content_bytes, media_type=mime_type))
                        elif mime_type == "application/pdf":
                            agent_input_list.append(f"[Content of PDF file '{filename}' was provided. Model should be prompted to process it if capable.]")
                            print(f"Warning: AnthropicAdapter with Pydantic model - PDF '{filename}' content sent as placeholder text to Agent.")
                        else:
                            agent_input_list.append(BinaryContent(data=content_bytes, media_type=mime_type))
                
                # Add text content from messages for Agent
                # This simplified loop takes string content from messages.
                # Anthropic's _format_messages_for_anthropic is more complex due to roles.
                # pydantic-ai Agent's run_sync expects a list of content parts.
                for msg in messages: # Using raw messages here
                    if isinstance(msg.get("content"), str):
                        agent_input_list.append(msg["content"])
                    # TODO: Handle complex content in messages if needed for Agent context

                extraction_prompt = (
                    f"Based on the preceding context and any provided files, "
                    f"extract the information and structure it strictly according to the '{pydantic_model_class.__name__}' Pydantic model. "
                    f"Ensure all fields are populated if information is available. "
                    f"If information for a field is not available, omit it or set it to null if the field is optional."
                )
                agent_input_list.append(extraction_prompt)

                if not agent_input_list: # Should not happen now
                    agent_input_list.append("Extract information based on the provided context.")

                system_instruction = (
                    f"You are an expert data extraction assistant. Your task is to extract information "
                    f"from the user's query and any provided context or files. Structure your response *strictly* "
                    f"according to the following Pydantic model: '{pydantic_model_class.__name__}'. "
                    f"Ensure all fields of the model are populated if the corresponding information is available. "
                    f"If information for a field is not found, and the field is optional, you may omit it or set it to null. "
                    f"Only output the Pydantic model instance as a JSON object, without any additional explanatory text or markdown."
                )

                # The AnthropicModel from pydantic-ai takes `model_name` and `provider`.
                # The `provider` defaults to 'anthropic', which should pick up env variables for API key.
                # It does not seem to take a `client` instance directly in its constructor.
                # It will create its own AsyncAnthropic client.
                anthropic_model_for_agent = AnthropicModel(
                    model_name=self.model_name
                    # provider can be specified if needed, but default 'anthropic' should work
                )
                agent = Agent(
                    model=anthropic_model_for_agent,
                    output_type=pydantic_model_class,
                    system_prompt=system_instruction,
                )
                
                result = agent.run_sync(agent_input_list)
                model_instance = result.output

                cost_info = {
                    "tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0,
                    "warning": "Token usage data not available via pydantic_ai.Agent for Anthropic in this version."
                }
                return {
                    "content": {"model_instance": model_instance, "text": None, "tool_calls": []},
                    "cost_info": cost_info
                }
            else:
                # Standard Anthropic API call
                anthropic_messages = self._format_messages_for_anthropic(messages, file_content_data)
                anthropic_tools = []
                if tools_details_aihelper:
                    for tool_schema in tools_details_aihelper:
                        func_schema = tool_schema.get("function", {})
                        anthropic_tools.append({
                            "name": func_schema.get("name"),
                            "description": func_schema.get("description"),
                            "input_schema": func_schema.get("parameters")
                        })
                
                api_params = {
                    "model": self.model_name,
                    "max_tokens": 4096, 
                    "messages": anthropic_messages,
                }
                if system_prompt: api_params["system"] = system_prompt
                if anthropic_tools: api_params["tools"] = anthropic_tools

                response = self.client.messages.create(**api_params)
                
                text_content = None
                tool_calls_result = []
                for block in response.content:
                    if block.type == "text":
                        text_content = (text_content or "") + block.text
                    elif block.type == "tool_use":
                        tool_calls_result.append({
                            "id": block.id, "type": "function",
                            "function": {"name": block.name, "arguments": json.dumps(block.input if block.input is not None else {})}
                        })
                
                if response.usage:
                    prompt_tokens = response.usage.input_tokens
                    completion_tokens = response.usage.output_tokens
                    total_tokens = prompt_tokens + completion_tokens
                    cost_info = {"tokens_used": total_tokens, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cost": 0}
                    if self.cost_tracker:
                        calculated_cost = self.cost_tracker.calculate_cost(self.model_name, prompt_tokens, completion_tokens)
                        if calculated_cost is not None: cost_info["cost"] = calculated_cost
                
                return {"content": {"text": text_content, "tool_calls": tool_calls_result}, "cost_info": cost_info}

        except APIError as e:
            print(f"Anthropic API Error: {e}")
            return {"content": {"text": f"Anthropic API Error: {e}", "model_instance": None, "tool_calls": []}, "cost_info": None}
        except Exception as e:
            print(f"Error during AnthropicAdapter process: {e}")
            return {"content": {"text": f"Adapter processing error: {e}", "model_instance": None, "tool_calls": []}, "cost_info": None}
