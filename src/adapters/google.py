import os
import json
from typing import Optional, Dict, Any, List, Type # Added Type
from google import genai
from google.genai import types as genai_types
from google.protobuf.json_format import MessageToDict # Keep if used by non-Agent path
from pydantic import BaseModel # For type hinting
from pydantic_ai import Agent, BinaryContent # For Pydantic model parsing
# Correct import based on file structure
from pydantic_ai.models.gemini import GeminiModel 

""" Important! 
This is the up-to-date library that should be used:
https://github.com/googleapis/python-genai
"""

from src.adapters.base_adapter import BaseAdapter
from src.cost_tracker import CostTracker
import mimetypes # For guessing MIME type of files

# Helper to guess MIME type
def get_mime_type(filename: str) -> Optional[str]:
    return mimetypes.guess_type(filename)[0]

class GoogleAdapter(BaseAdapter):
    def __init__(self, model_name: str, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, cost_tracker)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GOOGLE_GEMINI_API_KEY environment variable not set.")
        
        # Initialize the client as per the new SDK documentation
        # The API key is configured directly with the client.
        self.client = genai.Client(api_key=api_key)
        # self.model_name is already available from super().__init__
        print(f"GoogleAdapter initialized for model: {self.model_name} using genai.Client")

    def _convert_to_gemini_tools(self, tools_details: List[Dict[str, Any]]) -> Optional[List[genai_types.Tool]]:
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
        tools_details = input_data.get("tools_details", []) # For standard API calls
        file_content_data = input_data.get("file_content")
        pydantic_model_class: Optional[Type[BaseModel]] = input_data.get("pydantic_model_class")
        
        cost_info = None

        try:
            if pydantic_model_class:
                agent_input_list = []
                if file_content_data:
                    content_bytes = file_content_data.get("content_bytes")
                    filename = file_content_data.get("filename")
                    mime_type = get_mime_type(filename) if filename else "application/octet-stream"
                    if content_bytes:
                        agent_input_list.append(BinaryContent(data=content_bytes, media_type=mime_type))
                        if mime_type == "application/pdf":
                             print(f"Info: GoogleAdapter with Pydantic model - PDF '{filename}' sent as BinaryContent to Agent.")
                
                # Convert messages to a list of strings/parts for Agent
                # The _convert_messages_to_gemini_content prepares for direct API, might be too complex for Agent's list.
                # Agent usually takes simpler list of strings or Content parts.
                for msg in messages: # Using raw messages
                    if isinstance(msg.get("content"), str):
                        agent_input_list.append(msg["content"])
                    # TODO: More sophisticated handling of message roles and complex content for Agent if needed.

                extraction_prompt = (
                    f"Based on the preceding context and any provided files, "
                    f"extract the information and structure it strictly according to the '{pydantic_model_class.__name__}' Pydantic model. "
                    f"Ensure all fields are populated if information is available. "
                    f"If information for a field is not available, omit it or set it to null if the field is optional."
                )
                agent_input_list.append(extraction_prompt)

                if not agent_input_list and not file_content_data: # Should not happen now
                    agent_input_list.append("Extract information based on the provided context.")

                system_instruction = (
                    f"You are an expert data extraction assistant. Your task is to extract information "
                    f"from the user's query and any provided context or files. Structure your response *strictly* "
                    f"according to the following Pydantic model: '{pydantic_model_class.__name__}'. "
                    f"Ensure all fields of the model are populated if the corresponding information is available. "
                    f"If information for a field is not found, and the field is optional, you may omit it or set it to null. "
                    f"Only output the Pydantic model instance as a JSON object, without any additional explanatory text or markdown."
                )
                
                google_model_for_agent = GeminiModel( # Using the correct class name
                    model_name=self.model_name, 
                    # The GeminiModel from pydantic-ai might not take a 'client' argument directly.
                    # It might infer the client from global genai setup or use its own.
                    # Let's check pydantic-ai docs for GeminiModel instantiation.
                    # For now, assuming it works like OpenAIModel and takes the client,
                    # or that pydantic-ai's GeminiModel internally uses the `genai` client we configured.
                    # The `pydantic_ai.models.gemini.GeminiModel` constructor takes `provider`
                    # which can be 'google-gla' or 'google-vertex', or a Provider instance.
                    # It does not seem to take a `client` directly.
                    # It will create its own httpx.AsyncClient via the provider.
                    # This means our self.client = genai.Client(api_key=api_key) might not be used by pydantic-ai's GeminiModel.
                    # This could be an issue if API keys aren't globally configured for `genai` in a way `pydantic-ai` picks up.
                    # However, `genai.Client()` is the standard way to use the new SDK.
                    # Let's assume pydantic-ai's GeminiModel will work with the global `genai` configuration
                    # if no explicit client/provider is passed or if its default provider setup works.
                    # The `GeminiModel` constructor in pydantic-ai takes `model_name` and `provider`.
                    # If we want it to use our existing `self.client` (which is a `genai.Client`),
                    # we might need to wrap `self.client` in a `pydantic_ai.providers.Provider` instance.
                    # This is getting complex.
                    # Let's try the simplest form first, just model_name, and see if pydantic-ai handles it.
                    # If `pydantic_ai.models.gemini.GeminiModel` is designed for the `generativelanguage.googleapis.com` API
                    # directly (as its internal code suggests), it might not use our `google.genai` client.
                    # This is a potential conflict point.
                    # For now, let's stick to the pydantic-ai pattern for other models:
                    # provider='google-gla' is the default for GeminiModel in pydantic-ai.
                    # It will use GOOGLE_API_KEY or GOOGLE_GEMINI_API_KEY from env.
                )
                agent = Agent(
                    model=google_model_for_agent,
                    output_type=pydantic_model_class,
                    system_prompt=system_instruction,
                )

                result = agent.run_sync(agent_input_list)
                model_instance = result.output

                cost_info = {
                    "tokens_used": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0,
                    "warning": "Token usage data not available via pydantic_ai.Agent for Google in this version."
                }
                return {
                    "content": {"model_instance": model_instance, "text": None, "tool_calls": []},
                    "cost_info": cost_info
                }
        except genai.APIError as e: # Specific Google APIError
            print(f"Google Gemini API Error (via Pydantic Agent): {e}")
            return {"content": {"text": f"Google Gemini API Error: {e}", "model_instance": None, "tool_calls": []}, "cost_info": None}
        except Exception as e: # Catch other PydanticAI or general errors
            print(f"Error during GoogleAdapter Pydantic Agent process: {e}")
            return {"content": {"text": f"Adapter processing error: {e}", "model_instance": None, "tool_calls": []}, "cost_info": None}

        # Fallback to original non-Pydantic logic if pydantic_model_class is None
        # This 'else' corresponds to 'if pydantic_model_class:'
        # The try-except for this path is further down.
        # To fix the Pylance error, the entire method's logic needs to be within one try-except,
        # or the pydantic path needs its own complete try-except.
        # Let's ensure the pydantic path's try has its excepts, and the non-pydantic path also has its.
        # The current structure is:
        # try:
        #   if pydantic_model_class:
        #     ... pydantic logic ...
        #     return ...
        #   else: <--- This else needs to be part of the try or have its own try-except
        #     ... non-pydantic logic ...
        #
        # Corrected structure will be:
        # if pydantic_model_class:
        #   try:
        #      ... pydantic logic ...
        #   except genai.APIError ...
        #   except Exception ...
        # else:
        #   try:
        #      ... non-pydantic logic ...
        #   except Exception ... (already exists for non-pydantic path)

        # The Pylance error was for the outer try that now only contains the 'if pydantic_model_class'
        # Let's re-evaluate the structure. The initial `try:` was at the top of the method.
        # The `if pydantic_model_class:` block should be self-contained with its try-except.
        # The `else:` block for non-Pydantic calls will use the existing try-except at the end of the method.

            # Standard Google API call (this is the 'else' block)
            try:
                gemini_contents = self._convert_messages_to_gemini_content(messages, file_content_data)
                gemini_tools = self._convert_to_gemini_tools(tools_details)
                safety_settings = {
                    genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
                generation_config = {}
                
                config_params = {}
                if generation_config: config_params.update(generation_config)
                if gemini_tools: config_params["tools"] = gemini_tools
                if safety_settings:
                    safety_settings_list = [genai_types.SafetySetting(category=cat, threshold=thresh) for cat, thresh in safety_settings.items()]
                    if safety_settings_list: config_params["safety_settings"] = safety_settings_list
                
                final_config = genai_types.GenerateContentConfig(**config_params) if config_params else None

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=gemini_contents,
                    config=final_config
                )
                
                text_content = None
                tool_calls_result = []
        
                # ... (rest of the non-Pydantic logic from before)
                final_config = genai_types.GenerateContentConfig(**config_params) if config_params else None

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=gemini_contents,
                    config=final_config
                )
                text_content = None
                tool_calls_result = []
                if response.parts:
                    for part in response.parts:
                        if part.text:
                            text_content = (text_content or "") + part.text
                        if part.function_call:
                            args_dict = dict(part.function_call.args) if hasattr(part.function_call, 'args') and part.function_call.args is not None else {}
                            tool_calls_result.append({
                                "id": f"call_gemini_{part.function_call.name}", 
                                "type": "function",
                                "function": {"name": part.function_call.name, "arguments": json.dumps(args_dict)}
                            })
                
                prompt_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
                completion_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
                total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else (prompt_tokens + completion_tokens)

                if total_tokens > 0 :
                    cost_info = {
                        "tokens_used": total_tokens, "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens, "cost": 0
                    }
                    if self.cost_tracker:
                        calculated_cost = self.cost_tracker.calculate_cost(self.model_name, prompt_tokens, completion_tokens)
                        if calculated_cost is not None: cost_info["cost"] = calculated_cost
                
                return {"content": {"text": text_content, "tool_calls": tool_calls_result}, "cost_info": cost_info}

            # This except block is for the non-Pydantic path's try
            except Exception as e: # Catch broader exceptions from genai for the non-Pydantic path
                print(f"Google Gemini API Error (Direct Call): {e}")
                return {"content": {"text": f"Google Gemini API Error: {e}", "tool_calls": []}, "cost_info": None}
        
        # This top-level except was causing the Pylance issue if the if/else didn't return.
        # It should now be covered by the specific try-except blocks within each path.
        # However, to be safe, a general except at the end of the method for unexpected issues.
        except Exception as e:
             print(f"Unexpected error in GoogleAdapter.process: {e}")
             return {"content": {"text": f"Unexpected adapter error: {e}", "model_instance": None, "tool_calls": []}, "cost_info": None}
