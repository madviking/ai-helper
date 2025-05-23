from src.adapters.base_adapter import BaseAdapter
import os
import json
from dotenv import load_dotenv

load_dotenv()

class GoogleAdapter(BaseAdapter):
    def __init__(self, model_identifier: str = "gemini-1.5-pro"):
        self.model_identifier = model_identifier
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.client = genai
            self.model = genai.GenerativeModel(model_name=self.model_identifier)
        except ImportError:
            raise ImportError("google-genai library is not installed. Please install it with `pip install google-genai`")
        except Exception as e:
            raise Exception(f"Failed to initialize Google API: {str(e)}")

    def process(self, input_data):
        try:
            prompt = input_data.get("prompt", "")
            tools = input_data.get("tools", [])
            messages = [{"role": "user", "parts": [{"text": prompt}]}]
            
            # Handle file attachments if any
            if "files" in input_data:
                for file_data in input_data.get("files", []):
                    if isinstance(file_data, dict) and "path" in file_data:
                        file_path = file_data["path"]
                        if os.path.exists(file_path):
                            file_content = self.client.types.File(
                                uri=file_path,
                                mime_type=file_data.get("mime_type", "application/octet-stream")
                            )
                            messages[0]["parts"].append(file_content)
            
            # Configure tools if provided
            tool_config = None
            if tools:
                function_declarations = []
                for tool in tools:
                    if isinstance(tool, dict):
                        function_declarations.append({
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {})
                        })
                tool_config = self.client.types.ToolConfig(
                    function_calling=self.client.types.FunctionCallingConfig(
                        mode=self.client.types.FunctionCallingConfig.Mode.AUTO,
                        allowed_function_names=[tool.get("name") for tool in tools if isinstance(tool, dict)]
                    )
                )

            # Generate content
            response = self.model.generate_content(
                contents=messages,
                tools=tool_config
            )

            # Process response
            content = response.text if response.text else ""
            input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
            output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

            # Check for tool calls in the response
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        return {
                            "content": json.dumps({
                                "tool_call": True,
                                "tool_name": part.function_call.name,
                                "tool_args": {k: v for k, v in part.function_call.args.items()}
                            }),
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens
                        }

            return {
                "content": json.dumps({"content": content}),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        except Exception as e:
            return {
                "content": json.dumps({"error": str(e)}),
                "input_tokens": 0,
                "output_tokens": 0
            }
