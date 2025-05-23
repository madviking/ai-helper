import os
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError
import anthropic
from anthropic import Anthropic

from src.adapters.base_adapter import BaseAdapter


class AnthropicAdapter(BaseAdapter):
    def __init__(self):
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        self.client = Anthropic(api_key=api_key)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data using Anthropic API.
        
        Args:
            input_data: Dictionary containing:
                - prompt: The prompt to send
                - model: The model name
                - pydantic_model: Optional Pydantic model class for response validation
                - tools: Optional dictionary of available tools
                - file: Optional file path to include (not supported by Anthropic)
                
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
        
        # Anthropic doesn't support file input
        if file_path:
            raise ValueError("Anthropic models do not support file input")
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
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
                response = self.client.messages.create(
                    model=model,
                    messages=messages,
                    tools=tool_definitions,
                    max_tokens=4096
                )
            else:
                # Regular call without tools
                response = self.client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096
                )
            
            # Track tokens
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            
            # Handle tool use if present
            content = None
            for content_block in response.content:
                if content_block.type == 'text':
                    content = content_block.text
                elif content_block.type == 'tool_use' and tools:
                    # Execute tool
                    tool_name = content_block.name
                    tool_args = content_block.input
                    tool_result = tools[tool_name](**tool_args)
                    
                    # Continue conversation with tool result
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": json.dumps(tool_result)
                        }]
                    })
                    
                    # Get final response
                    final_response = self.client.messages.create(
                        model=model,
                        messages=messages,
                        max_tokens=4096
                    )
                    
                    total_input_tokens += final_response.usage.input_tokens
                    total_output_tokens += final_response.usage.output_tokens
                    
                    # Extract text content from final response
                    for block in final_response.content:
                        if block.type == 'text':
                            content = block.text
                            break
            
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
            raise Exception(f"Anthropic API error: {str(e)}")

    def _prepare_tool_definitions(self, tools: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare tool definitions for Anthropic API."""
        tool_defs = []
        
        for tool_name, tool_func in tools.items():
            # Extract function signature and docstring
            import inspect
            sig = inspect.signature(tool_func)
            doc = inspect.getdoc(tool_func) or f"Function {tool_name}"
            
            # Build input schema
            input_schema = {
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
                
                input_schema["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
                
                # Check if required
                if param.default == param.empty:
                    input_schema["required"].append(param_name)
            
            tool_defs.append({
                "name": tool_name,
                "description": doc,
                "input_schema": input_schema
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
