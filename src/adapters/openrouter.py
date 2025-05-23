import os
import json
import base64
import mimetypes
from typing import Optional, Dict, Any, List, Type

from openai import OpenAI, APIError # For OpenRouter client
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent # For Pydantic model parsing

from src.adapters.base_adapter import BaseAdapter
from src.cost_tracker import CostTracker

# Helper from openai_adapter, can be moved to a common utils if used by more
def get_mime_type(filename: str) -> Optional[str]:
    return mimetypes.guess_type(filename)[0]

class OpenRouterAdapter(BaseAdapter):
    def __init__(self, model_name: str, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, cost_tracker)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
        
        self.client = OpenAI(
          base_url="https://openrouter.ai/api/v1",
          api_key=api_key,
        )
        # Headers can be added here if needed, e.g., for site identification
        # self.client.default_headers["HTTP-Referer"] = "YOUR_SITE_URL" 
        # self.client.default_headers["X-Title"] = "YOUR_SITE_NAME"
        print(f"OpenRouterAdapter initialized for model: {self.model_name}")

    # Reusing _format_messages_for_openai from OpenAIAdapter as OpenRouter is OpenAI-compatible
    # This assumes _format_messages_for_openai is defined in a way that can be reused or adapted.
    # For simplicity, let's copy a simplified version here or call it if it's in a shared util.
    # For now, we'll assume a similar formatting logic as OpenAIAdapter's internal one.
    def _format_messages_for_openrouter(self, messages: List[Dict[str, Any]], file_content_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This is a simplified version. A more robust solution would share/inherit this.
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")

            or_msg = {"role": role}
            if role == "user" and i == 0 and file_content_data:
                filename = file_content_data.get("filename")
                content_bytes = file_content_data.get("content_bytes")
                mime_type = get_mime_type(filename) if filename else None
                if mime_type and mime_type.startswith("image/") and content_bytes: # Basic image handling
                    base64_image = base64.b64encode(content_bytes).decode('utf-8')
                    image_url = f"data:{mime_type};base64,{base64_image}"
                    multi_content = [{"type": "text", "text": content if isinstance(content, str) else ""}]
                    multi_content.append({"type": "image_url", "image_url": {"url": image_url}})
                    or_msg["content"] = multi_content
                elif isinstance(content, str):
                    or_msg["content"] = content
                else:
                    or_msg["content"] = ""
            elif isinstance(content, str):
                or_msg["content"] = content
            
            if tool_calls: or_msg["tool_calls"] = tool_calls
            if role == "tool" and tool_call_id:
                or_msg["tool_call_id"] = tool_call_id
                or_msg["name"] = name
                or_msg["content"] = str(content) if content is not None else ""
            
            if "content" not in or_msg and role == "assistant" and not tool_calls: or_msg["content"] = ""
            elif "content" not in or_msg and role != "tool" and role != "assistant": or_msg["content"] = str(content) if content is not None else ""
            
            if role == "assistant" and not or_msg.get("content") and not or_msg.get("tool_calls"): continue
            formatted_messages.append(or_msg)
        return formatted_messages

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = input_data.get("messages", [])
        tools_details = input_data.get("tools_details", [])
        file_content_data = input_data.get("file_content")
        pydantic_model_class: Optional[Type[BaseModel]] = input_data.get("pydantic_model_class")

        # Use the same message formatting as OpenAI for OpenRouter
        # This assumes OpenRouter models behave like OpenAI models regarding message structure.
        # The _format_messages_for_openai method from OpenAIAdapter could be moved to a shared utility
        # or BaseAdapter if it's generic enough. For now, using the local simplified version.
        processed_messages = self._format_messages_for_openrouter(messages, file_content_data)
        
        cost_info = None

        try:
            if pydantic_model_class:
                # Use pydantic_ai.Agent. The model_identifier for Agent needs to be OpenAI-compatible
                # even if the actual model_name (self.model_name) is an OpenRouter specific string.
                # pydantic-ai's OpenAI model integration will use the self.client (configured for OpenRouter).
                # The key is that the *client* is OpenAI compatible.
                # We might need to tell Agent it's an "openai" type model, but pass the OpenRouter model name
                # to the client it uses.
                # The `Agent` constructor takes `model_identifier` which can be 'provider:model_name'.
                # For OpenRouter, if it uses an OpenAI client, we might use 'openai:actual_model_name_on_openrouter'
                # or rely on pydantic-ai to use the client's model if not specified in identifier.
                # Let's assume pydantic-ai Agent can use the client configured for OpenRouter
                # and we pass the OpenRouter model name (self.model_name) to the underlying client call
                # via pydantic-ai's mechanisms.
                # The `agent_model_identifier` for pydantic-ai Agent should be 'openai:model_name'
                # to signal it to use its OpenAI integration, which will then use our custom client.
                
                # Construct input for agent.run_sync
                agent_input_list = []
                if file_content_data:
                    content_bytes = file_content_data.get("content_bytes")
                    filename = file_content_data.get("filename")
                    mime_type = get_mime_type(filename) if filename else "application/octet-stream"
                    if content_bytes:
                        if mime_type and mime_type.startswith("image/"):
                             agent_input_list.append(BinaryContent(data=content_bytes, media_type=mime_type))
                        elif mime_type == "application/pdf":
                            # OpenRouter models might support PDF differently or not at all via this path.
                            # For now, send placeholder text.
                            agent_input_list.append(f"[Content of PDF file '{filename}' was provided. Model should be prompted to process it if capable.]")
                            print(f"Warning: OpenRouterAdapter with Pydantic model - PDF '{filename}' content sent as placeholder text to Agent.")
                        else: # Other file types as BinaryContent
                            agent_input_list.append(BinaryContent(data=content_bytes, media_type=mime_type))

                for msg_dict in processed_messages:
                    if isinstance(msg_dict.get("content"), str):
                        agent_input_list.append(msg_dict["content"])
                    elif isinstance(msg_dict.get("content"), list):
                        for content_part in msg_dict["content"]:
                            if content_part.get("type") == "text":
                                agent_input_list.append(content_part["text"])
                
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

                # For pydantic-ai with OpenRouter, we tell the Agent it's an "openai" model type,
                # and the Agent will use the client provided to its OpenAIModel, which is our
                # OpenRouter-configured OpenAI client. The actual model name (self.model_name)
                # will be passed to the API call by the client.
                from pydantic_ai.models import OpenAIModel
                openai_model_for_agent = OpenAIModel(model=self.model_name, client=self.client)

                agent = Agent(
                    model=openai_model_for_agent, # Pass the configured OpenAIModel
                    output_type=pydantic_model_class,
                    system_prompt=system_instruction,
                )
                
                result = agent.run_sync(agent_input_list)
                model_instance = result.output
                
                # Placeholder for cost tracking with pydantic_ai and OpenRouter
                cost_info = {
                    "tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0,
                    "warning": "Token usage data not available via pydantic_ai.Agent for OpenRouter in this version."
                }
                return {
                    "content": {"model_instance": model_instance, "text": None, "tool_calls": []},
                    "cost_info": cost_info
                }
            else:
                # Standard OpenRouter call (OpenAI compatible)
                api_params = {
                    "model": self.model_name, # OpenRouter model string e.g. "google/gemini-pro"
                    "messages": processed_messages,
                }
                if tools_details:
                    api_params["tools"] = tools_details
                    api_params["tool_choice"] = "auto" # Or handle specific tool choice

                completion = self.client.chat.completions.create(**api_params)
                response_message = completion.choices[0].message
                
                text_content = response_message.content
                tool_calls_data = []
                if response_message.tool_calls:
                    tool_calls_data = [
                        {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in response_message.tool_calls
                    ]
                    text_content = None # If there are tool_calls, text_content is usually None

                # Cost tracking for OpenRouter
                # OpenRouter API response includes an 'X-OpenRouter-Cost' header or usage in body for some models.
                # The OpenAI Python client might not directly expose custom headers easily.
                # For now, using usage data if available, similar to OpenAI.
                if completion.usage:
                    cost_info = {
                        "tokens_used": completion.usage.total_tokens,
                        "prompt_tokens": completion.usage.prompt_tokens,
                        "completion_tokens": completion.usage.completion_tokens,
                        "cost": 0 # Placeholder, OpenRouter cost varies
                    }
                    if self.cost_tracker:
                        # Cost calculation for OpenRouter needs specific pricing for self.model_name
                        # This might require fetching OpenRouter's pricing or having it pre-loaded.
                        # For now, if cost_tracker has a method, use it, otherwise cost is 0.
                        calculated_cost = self.cost_tracker.calculate_cost(
                            self.model_name, # Pass the OpenRouter model name
                            completion.usage.prompt_tokens,
                            completion.usage.completion_tokens
                        )
                        if calculated_cost is not None:
                            cost_info["cost"] = calculated_cost
                        else:
                             cost_info["warning"] = f"Cost for OpenRouter model {self.model_name} not found in pricing."
                else: # Fallback if no usage data
                    cost_info = {"tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0, "warning": "Usage data not available from OpenRouter response."}


                return {
                    "content": {"text": text_content, "tool_calls": tool_calls_data},
                    "cost_info": cost_info
                }

        except APIError as e:
            print(f"OpenRouter API Error (via OpenAI client): {e}")
            return {"content": {"text": f"OpenRouter API Error: {e}", "model_instance": None, "tool_calls": []}, "cost_info": None}
        except Exception as e:
            print(f"Error during OpenRouterAdapter process: {e}")
            return {"content": {"text": f"Adapter processing error: {e}", "model_instance": None, "tool_calls": []}, "cost_info": None}
