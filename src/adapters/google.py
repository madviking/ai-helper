import os
import json
from typing import Optional, Dict, Any, List
import google.generativeai as genai # Reverting to this based on common SDK usage
from google.generativeai import types as genai_types
from google.protobuf.json_format import MessageToDict


from src.adapters.base_adapter import BaseAdapter
from src.cost_tracker import CostTracker
import mimetypes # For guessing MIME type of files

# Helper to guess MIME type
def get_mime_type(filename: str) -> Optional[str]:
    return mimetypes.guess_type(filename)[0]

class GoogleAdapter(BaseAdapter):
    def __init__(self, model_name: str, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, cost_tracker)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Attempt to use GOOGLE_GEMINI_API_KEY if GOOGLE_API_KEY is not set, common for Gemini specific keys
            api_key = os.getenv("GOOGLE_GEMINI_API_KEY") 
            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GOOGLE_GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        # For Gemini, model_name might just be e.g. "gemini-1.5-flash" or "gemini-pro"
        # The SDK prepends "models/" if not already present for some calls.
        # We'll assume self.model_name is the direct model identifier like "gemini-1.5-flash-latest".
        self.client = genai.GenerativeModel(model_name=self.model_name) # Use model_name=
        print(f"GoogleAdapter initialized for model: {self.model_name}")

    def _convert_to_gemini_tools(self, tools_details: List[Dict[str, Any]]) -> Optional[List[genai_types.Tool]]: # Prefixed Tool
        if not tools_details:
            return None
        
        gemini_tools = []
        for tool_detail in tools_details:
            func_detail = tool_detail.get("function", {})
            name = func_detail.get("name")
            description = func_detail.get("description")
            parameters_schema = func_detail.get("parameters") # This is already a JSON schema object

            if not name or not description or not parameters_schema:
                print(f"Warning: Skipping tool due to missing details: {name}")
                continue

            # Gemini expects FunctionDeclaration
            func_decl = genai_types.FunctionDeclaration( 
                name=name,
                description=description,
                parameters=parameters_schema
            )
            gemini_tools.append(genai_types.Tool(function_declarations=[func_decl]))
        return gemini_tools if gemini_tools else None

    def _convert_messages_to_gemini_content(self, messages: List[Dict[str, Any]], file_content_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        gemini_contents = []
        for i, message in enumerate(messages):
            role = message.get("role")
            # Gemini uses 'user' and 'model' roles. 'system' prompts are handled differently.
            # 'assistant' maps to 'model'. 'tool' maps to a function response part.
            gemini_role = "user" if role == "user" else "model"

            parts = []
            
            # Handle text content
            text_content = message.get("content")
            if isinstance(text_content, str) and text_content:
                parts.append(genai_types.Part(text=text_content))

            # Handle file input for the first user message
            if role == "user" and i == 0 and file_content_data:
                filename = file_content_data.get("filename")
                content_bytes = file_content_data.get("content_bytes")
                mime_type = get_mime_type(filename) if filename else "application/octet-stream"
                
                if content_bytes and mime_type:
                    parts.append(genai_types.Part(inline_data={"mime_type": mime_type, "data": content_bytes}))
            
            # Handle tool calls made by the model (assistant role)
            if role == "assistant" and message.get("tool_calls"):
                for tool_call in message.get("tool_calls", []):
                    func_call_data = tool_call.get("function", {})
                    # Gemini expects arguments as a dict, not a JSON string
                    args_dict = {}
                    try:
                        args_dict = json.loads(func_call_data.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse tool arguments for {func_call_data.get('name')}")

                    parts.append(genai_types.Part(function_call={"name": func_call_data.get("name"), "args": args_dict}))
            
            # Handle tool responses (tool role)
            if role == "tool":
                gemini_role = "model" # Function responses are from the 'model' (acting as a function/tool)
                tool_name = message.get("name")
                tool_output_content = str(message.get("content", "")) # Ensure it's a string
                
                # Gemini expects a FunctionResponse part
                parts.append(genai_types.Part(function_response={"name": tool_name, "response": {"content": tool_output_content}}))


            if parts: # Only add if there are parts to send
                 # For Gemini, 'function' role parts (tool responses) should be part of a 'model' turn
                 # or handled by constructing the history carefully.
                 # If the current role is 'tool', we've made it a 'model' role with a function_response part.
                gemini_contents.append({"role": gemini_role, "parts": parts})
        
        return gemini_contents


    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        messages = input_data.get("messages", [])
        tools_details = input_data.get("tools_details", [])
        file_content_data = input_data.get("file_content")
        # pydantic_model_json_schema = input_data.get("pydantic_model_json_schema") # For JSON mode

        gemini_contents = self._convert_messages_to_gemini_content(messages, file_content_data)
        gemini_tools = self._convert_to_gemini_tools(tools_details)

        # Safety settings (optional, but good practice)
        safety_settings = {
            genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        generation_config = {}
        # TODO: if pydantic_model_json_schema: generation_config["response_mime_type"] = "application/json"
        # This requires the model to support JSON output mode.

        try:
            response = self.client.generate_content(
                contents=gemini_contents,
                tools=gemini_tools if gemini_tools else None, # Pass None if no tools
                generation_config=genai_types.GenerationConfig(**generation_config) if generation_config else None,
                safety_settings=safety_settings
            )
        except Exception as e: # Catch broader exceptions from genai
            print(f"Google Gemini API Error: {e}")
            return {"content": {"text": f"Google Gemini API Error: {e}", "tool_calls": []}, "cost_info": None}

        text_content = None
        tool_calls_result = []
        
        # Process response parts
        if response.parts:
            for part in response.parts:
                if part.text:
                    text_content = (text_content or "") + part.text
                if part.function_call:
                    # Convert arguments from Struct to dict, then to JSON string for consistency with OpenAI
                    args_dict = MessageToDict(part.function_call.args) if hasattr(part.function_call, 'args') else {}
                    tool_calls_result.append({
                        # Gemini doesn't provide a tool_call_id in the same way OpenAI does.
                        # We might need to generate one or adapt AiHelper's expectations.
                        # For now, using the function name as a placeholder ID if needed by AiHelper.
                        "id": f"call_gemini_{part.function_call.name}", 
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(args_dict)
                        }
                    })
        
        cost_info = None
        # Gemini API usually returns token counts in response.usage_metadata (if available)
        # or one might need to call count_tokens separately.
        # For now, this is a placeholder.
        prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
        completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else (prompt_tokens + completion_tokens)

        if total_tokens > 0 : # Only create cost_info if we have token counts
            cost_info = {
                "tokens_used": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": 0 # Placeholder
            }
            if self.cost_tracker:
                calculated_cost = self.cost_tracker.calculate_cost(
                    self.model_name, prompt_tokens, completion_tokens
                )
                if calculated_cost is not None:
                    cost_info["cost"] = calculated_cost
        
        return {
            "content": {"text": text_content, "tool_calls": tool_calls_result},
            "cost_info": cost_info
        }
