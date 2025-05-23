from src.adapters.base_adapter import BaseAdapter
from anthropic import Anthropic
import os
import json
from dotenv import load_dotenv

load_dotenv()

class AnthropicAdapter(BaseAdapter):
    def __init__(self, model_identifier: str = "claude-3-opus-20240229"):
        self.model_identifier = model_identifier
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def process(self, input_data):
        prompt = input_data.get("prompt", "")
        tools = input_data.get("tools", [])
        messages = [{"role": "user", "content": prompt}]
        
        # Handle image or file if provided
        if "image" in input_data:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": input_data['image']}}
            ]
        elif "file" in input_data:
            messages[0]["content"] = f"{prompt}\n\nFile content reference: {input_data['file']}"
        
        try:
            # Prepare tools for function calling if provided
            tools_param = []
            if tools:
                for tool in tools:
                    if isinstance(tool, dict):
                        tools_param.append({
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "input_schema": tool.get("parameters", {})
                        })
            
            response = self.client.messages.create(
                model=self.model_identifier,
                max_tokens=1024,
                messages=messages,
                tools=tools_param if tools_param else None,
                temperature=0.7
            )
            
            content = {}
            if response.content and len(response.content) > 0:
                content_block = response.content[0]
                if content_block.type == "tool_use":
                    content = {
                        "tool_call": True,
                        "tool_name": content_block.name,
                        "tool_args": content_block.input if content_block.input else {}
                    }
                else:
                    content = {"content": content_block.text if content_block.text else ""}
                    if "extracted_data" in input_data:
                        content["extracted_data"] = "Extracted data placeholder"
            
            return {
                "content": json.dumps(content),
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0
            }
        except Exception as e:
            error_content = {"content": f"Error processing request: {str(e)}"}
            return {
                "content": json.dumps(error_content),
                "input_tokens": 0,
                "output_tokens": 0
            }
