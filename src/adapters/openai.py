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
                
                # Extract the last user message content as the prompt for the Agent.
                # This is a simplification; a more robust solution might concatenate messages
                # or require pydantic-ai's Agent to handle a list of messages.
                prompt_for_agent = "Extract information." # Default prompt
                if openai_messages:
                    # Find the last user message, or the first if only one.
                    # PydanticAI Agent might take the 'messages' kwarg.
                    # Let's assume it takes a simple prompt string for `run_sync` based on the example.
                    last_user_message_content = ""
                    for msg in reversed(openai_messages):
                        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                            last_user_message_content = msg["content"]
                            break
                    if last_user_message_content:
                         prompt_for_agent = last_user_message_content
                    elif isinstance(openai_messages[0].get("content"), str) : # Fallback to first message if no user message or complex content
                        prompt_for_agent = openai_messages[0]["content"]


                # The OpenAI example for pydantic-ai shows passing a `model` instance of `OpenAIResponsesModel`
                # and `model_settings` to the Agent.
                # from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
                # model_settings = OpenAIResponsesModelSettings(...)
                # model = OpenAIResponsesModel(self.model_name) # Pass model name string
                # agent = Agent(model=model, model_settings=model_settings, output_type=pydantic_model_class)
                # This seems more aligned with the OpenAI-specific example provided.

                from pydantic_ai.models.openai import OpenAIResponsesModel #, OpenAIResponsesModelSettings
                
                # We don't have specific model_settings like web search tools for this general case.
                # So, we might just need OpenAIResponsesModel.
                # The Agent constructor in the example is: Agent(model=model, model_settings=model_settings)
                # Or for simpler cases: Agent('openai:gpt-4o', output_type=SupportOutput)

                # Let's try the simpler Agent initialization first, then refine if needed.
                # The first example was: agent = Agent('google-gla:gemini-1.5-flash', ...)
                # So for OpenAI:
                agent = Agent(
                    agent_model_identifier, # e.g., 'openai:gpt-3.5-turbo'
                    output_type=pydantic_model_class,
                    # system_prompt can be added if needed
                )
                
                # The run_sync method in the example takes a single string.
                # TODO: Investigate if pydantic-ai Agent can take a list of messages.
                # For now, using the prompt_for_agent derived above.
                result = agent.run_sync(prompt_for_agent) 
                
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
