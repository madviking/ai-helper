import os
import json
import base64
import mimetypes
from typing import Optional, Dict, Any, List, Type
from openai import OpenAI, APIError # Keep for non-agent usage or if Agent needs it
from pydantic import BaseModel
from pydantic_ai import Agent # Import Agent
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
        # Standard OpenAI client, might be used by Agent or for non-Pydantic calls
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
        tools_details = input_data.get("tools_details", [])
        file_content_data = input_data.get("file_content")
        pydantic_model_class: Optional[Type[BaseModel]] = input_data.get("pydantic_model_class")

        openai_messages = self._format_messages_for_openai(messages, file_content_data)
        
        cost_info = None # Initialize cost_info

        try:
            if pydantic_model_class:
                # Use pydantic_ai.Agent for structured output
                agent_model_identifier = f"openai:{self.model_name}"
                
                # The Agent needs a system prompt or a user prompt.
                # Let's try to pass the messages to the agent if possible,
                # or construct a single prompt string from the last user message.
                # The example `agent.run_sync('Where does "hello world" come from?')` suggests a single string.
                
                # Construct the input for agent.run_sync based on pydantic-ai documentation
                agent_input_list = []
                
                # Add formatted messages (text parts)
                # The _format_messages_for_openai currently creates a list of dicts.
                # We need to adapt this or extract content for the agent_input_list.
                # For now, let's take the text content from the messages.
                # A more sophisticated approach would map roles and complex content types if Agent supports them.
                # The pydantic-ai examples show a list of strings and ImageUrl/BinaryContent.
                
                # Simplified: add text from user messages and the last assistant message if any.
                # And add file content.
                
                # Add file content first if present
                if file_content_data:
                    content_bytes = file_content_data.get("content_bytes")
                    filename = file_content_data.get("filename")
                    mime_type = get_mime_type(filename) if filename else "application/octet-stream"

                    if content_bytes:
                        from pydantic_ai import BinaryContent # Import locally
                        if mime_type and mime_type.startswith("image/"):
                            agent_input_list.append(BinaryContent(data=content_bytes, media_type=mime_type))
                        elif mime_type == "application/pdf":
                            # pydantic-ai's BinaryContent with application/pdf seems to cause errors with OpenAI Chat Completions.
                            # For now, indicate a PDF was present but don't send its raw binary via Agent this way.
                            agent_input_list.append(f"[Content of PDF file '{filename}' was provided but not directly processed by Agent for OpenAI due to API limitations. Consider summarizing or querying its content via text.]")
                            print(f"Warning: OpenAIAdapter with Pydantic model - PDF '{filename}' content not sent directly to Agent. Text extraction or a different API approach might be needed for OpenAI to process PDF content for Pydantic output.")
                        else: # Other file types
                            agent_input_list.append(BinaryContent(data=content_bytes, media_type=mime_type))
                
                # Add text content from messages
                # This needs to be a sequence of prompts/contexts.
                # Let's combine user messages and the last assistant message (if any) into a context string for now.
                # Or, pass each message's content as a string if the Agent handles a list of strings sequentially.
                # The example `agent.run_sync(['What company is this logo from?', ImageUrl(...)])` suggests it does.

                for msg_dict in openai_messages: # openai_messages is already formatted
                    if isinstance(msg_dict.get("content"), str):
                        agent_input_list.append(msg_dict["content"])
                    elif isinstance(msg_dict.get("content"), list): # For multimodal user messages
                        for content_part in msg_dict["content"]:
                            if content_part.get("type") == "text":
                                agent_input_list.append(content_part["text"])
                            # ImageUrl is handled by BinaryContent above if it's from file_content_data
                            # If ImageUrl was part of messages directly, it would need handling here.
                
                # Add a specific instruction for Pydantic model extraction
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

                agent = Agent(
                    agent_model_identifier, 
                    output_type=pydantic_model_class,
                    system_prompt=system_instruction,
                )
                
                result = agent.run_sync(agent_input_list) 
                model_instance = result.output

                # COST TRACKING: The provided examples for pydantic_ai.Agent
                # do NOT show how to get token usage. This is a significant issue.
                # For now, cost_info will be a placeholder.
                cost_info = {
                    "tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0,
                    "warning": "Token usage data not available via pydantic_ai.Agent in this version."
                }
                # If self.cost_tracker exists, it won't be able to calculate cost without token counts.

                return {
                    "content": {"model_instance": model_instance, "text": None, "tool_calls": []},
                    "cost_info": cost_info
                }
            else:
                # Standard processing using the raw OpenAI client (self.client)
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
