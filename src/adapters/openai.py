from src.adapters.base_adapter import BaseAdapter
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model_identifier: str = "gpt-4-turbo"):
        self.model_identifier = model_identifier
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def process(self, input_data):
        prompt = input_data.get("prompt", "")
        tools = input_data.get("tools", [])
        messages = [{"role": "user", "content": prompt}]
        
        # Handle image or file if provided
        if "image" in input_data:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{input_data['image']}"}}]
        elif "file" in input_data:
            # For files, we might need to upload and reference them, but for simplicity, include in prompt
            messages[0]["content"] = f"{prompt}\n\nFile content reference: {input_data['file']}"
        
        try:
            # Prepare tools for function calling if provided
            tools_param = []
            if tools:
                for tool in tools:
                    if isinstance(tool, dict):
                        tools_param.append({
                            "type": "function",
                            "function": {
                                "name": tool.get("name", ""),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("parameters", {})
                            }
                        })
            
            response = self.client.chat.completions.create(
                model=self.model_identifier,
                messages=messages,
                tools=tools_param if tools_param else None,
                tool_choice="auto" if tools_param else None,
                temperature=0.7
            )
            
            message = response.choices[0].message
            content = {}
            
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                content = {
                    "tool_call": True,
                    "tool_name": tool_call.function.name,
                    "tool_args": json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                }
            else:
                content = {"content": message.content}
                if "extracted_data" in input_data:
                    content["extracted_data"] = "Extracted data placeholder"
            
            return {
                "content": json.dumps(content),
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0
            }
        except Exception as e:
            error_content = {"content": f"Error processing request: {str(e)}"}
            return {
                "content": json.dumps(error_content),
                "input_tokens": 0,
                "output_tokens": 0
            }
