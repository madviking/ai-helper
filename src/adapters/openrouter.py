import os
import json
import base64
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError
from openai import OpenAI

from src.adapters.base_adapter import BaseAdapter


class OpenRouterAdapter(BaseAdapter):
    def __init__(self):
        api_key = os.environ.get('OPENROUTER_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_KEY environment variable is not set")
        
        # OpenRouter uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://openrouter.ai/api/v1',
            default_headers={
                'HTTP-Referer': 'https://github.com/ai-helper',
                'X-Title': 'AI Helper'
            }
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data using OpenRouter API.
        
        Args:
            input_data: Dictionary containing:
                - prompt: The prompt to send
                - model: The model name
                - pydantic_model: Optional Pydantic model class for response validation
                - tools: Optional dictionary of available tools
                - file: Optional file path to include
                
        Returns:
            Dictionary containing:
                - response: Pydantic model instance or raw response
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - model_used: The model that was used
        """
        prompt = input_data['prompt']
        model = input_data['model']
        pydantic_model = input_data.get('pydantic_model')
        tools = input_data.get('tools')
        file_path = input_data.get('file')
        
        # Map model names to OpenRouter format if needed
        model = self._map_model_name(model)
        
        # Check if model supports file input
        if file_path and model.startswith('anthropic/'):
            raise ValueError(f"Model '{model}' does not support file input")
        
        # Prepare messages
        messages = []
        
        # Handle file input
        if file_path:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                # Handle image file
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._encode_image(file_path)}}
                    ]
                })
            else:
                # For other files, read as text
                file_content = self._read_file_content(file_path)
                messages.append({
                    "role": "user",
                    "content": f"{prompt}\n\nFile content:\n{file_content}"
                })
        else:
            messages.append({"role": "user", "content": prompt})
        
        # Prepare tools if available
        tool_definitions = None
        if tools:
            tool_definitions = self._prepare_tool_definitions(tools)
        
        # Make API call
        total_input_tokens = 0
        total_output_tokens = 0
        
        try:
            if tool_definitions:
                # Call with tools
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_definitions,
                    tool_choice="auto"
                )
            else:
                # Regular call without tools
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages
                )
            
            # Track tokens
            total_input_tokens += response.usage.prompt_tokens
            total_output_tokens += response.usage.completion_tokens
            
            # Handle tool calls if present
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Execute tools and get final response
                messages.append(message)
                
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    # Execute tool
                    tool_result = tools[tool_name](**tool_args)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })
                
                # Get final response
                final_response = self.client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                
                total_input_tokens += final_response.usage.prompt_tokens
                total_output_tokens += final_response.usage.completion_tokens
                
                content = final_response.choices[0].message.content
            else:
                content = message.content
            
            # Parse response based on pydantic_model
            if pydantic_model:
                response_obj = self._parse_pydantic_response(content, pydantic_model)
            else:
                response_obj = content
            
            return {
                'response': response_obj,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'model_used': model
            }
            
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")

    def _map_model_name(self, model: str) -> str:
        """Map model names to OpenRouter format."""
        # If already in provider/model format, return as is
        if '/' in model:
            return model
        
        # Map common model names to provider/model format
        model_mappings = {
            # OpenAI models
            'gpt-3.5-turbo': 'openai/gpt-3.5-turbo',
            'gpt-4': 'openai/gpt-4',
            'gpt-4-turbo': 'openai/gpt-4-turbo',
            'gpt-4-vision-preview': 'openai/gpt-4-vision-preview',
            
            # Anthropic models
            'claude-3': 'anthropic/claude-3',
            'claude-3-opus': 'anthropic/claude-3-opus',
            'claude-3-sonnet': 'anthropic/claude-3-sonnet',
            'claude-3-haiku': 'anthropic/claude-3-haiku',
            'claude-3-opus-20240229': 'anthropic/claude-3-opus-20240229',
            
            # Google models
            'gemini-2.5-flash-preview-05-20': 'google/gemini-2.5-flash-preview-05-20',
            'gemini-pro': 'google/gemini-pro',
            'gemini-pro-vision': 'google/gemini-pro-vision',
        }
        
        return model_mappings.get(model, model)

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 data URL."""
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine mime type
        ext = image_path.lower().split('.')[-1]
        mime_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{encoded}"

    def _read_file_content(self, file_path: str) -> str:
        """Read file content as text."""
        if file_path.lower().endswith('.pdf'):
            # Use pdf_reader tool if available
            from src.tools import pdf_reader
            return pdf_reader(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _prepare_tool_definitions(self, tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare tool definitions for OpenRouter API (OpenAI format)."""
        tool_defs = []
        
        for tool_name, tool_func in tools.items():
            # Extract function signature and docstring
            import inspect
            sig = inspect.signature(tool_func)
            doc = inspect.getdoc(tool_func) or f"Function {tool_name}"
            
            # Build parameters schema
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                # Determine parameter type
                param_type = "string"  # default
                if param.annotation != param.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                
                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
                
                # Check if required
                if param.default == param.empty:
                    parameters["required"].append(param_name)
            
            tool_defs.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": doc,
                    "parameters": parameters
                }
            })
        
        return tool_defs

    def _parse_pydantic_response(self, content: str, pydantic_model: type[BaseModel]) -> BaseModel:
        """Parse response content into Pydantic model."""
        try:
            # Try to parse as JSON
            data = json.loads(content)
        except json.JSONDecodeError:
            # If not JSON, try to extract JSON from the content
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}
        
        # Create instance with partial validation
        instance = pydantic_model()
        valid_fields = 0
        total_fields = len(pydantic_model.model_fields)
        
        for field_name, field_info in pydantic_model.model_fields.items():
            if field_name in data:
                try:
                    # Validate individual field
                    value = data[field_name]
                    # Create a temporary model with just this field to validate
                    temp_model = type('TempModel', (BaseModel,), {field_name: field_info})
                    temp_instance = temp_model(**{field_name: value})
                    setattr(instance, field_name, getattr(temp_instance, field_name))
                    valid_fields += 1
                except (ValidationError, ValueError, TypeError):
                    # Field validation failed, keep default value
                    pass
        
        # Calculate fill percentage
        if total_fields > 0:
            fill_percentage = (valid_fields / total_fields) * 100
        else:
            fill_percentage = 100
        
        # Store fill percentage as a non-field attribute
        instance._fill_percentage = fill_percentage
        
        return instance
