import os
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError
import google.generativeai as genai
from PIL import Image

from src.adapters.base_adapter import BaseAdapter


class GoogleAdapter(BaseAdapter):
    def __init__(self):
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data using Google Generative AI API.
        
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
        model_name = input_data['model']
        pydantic_model = input_data.get('pydantic_model')
        tools = input_data.get('tools')
        file_path = input_data.get('file')
        
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        
        # Prepare content
        content_parts = []
        
        # Add prompt
        content_parts.append(prompt)
        
        # Handle file input
        if file_path:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                # Handle image file
                image = Image.open(file_path)
                content_parts.append(image)
            else:
                # For other files, read as text
                file_content = self._read_file_content(file_path)
                content_parts.append(f"\n\nFile content:\n{file_content}")
        
        # Prepare tools if available
        if tools:
            # Google AI doesn't support function calling in the same way
            # We'll need to handle this differently or skip for now
            # For now, we'll just mention available tools in the prompt
            tools_desc = self._describe_tools(tools)
            content_parts.insert(0, f"Available tools:\n{tools_desc}\n\nIf you need to use any tool, respond with a JSON object containing 'tool_name' and 'tool_args'.\n\n")
        
        try:
            # Generate content
            response = model.generate_content(content_parts)
            
            # Extract token usage
            # Note: Google AI doesn't provide token counts in the same way
            # We'll estimate based on content length
            input_tokens = self._estimate_tokens(prompt + (file_content if file_path and not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) else ''))
            output_tokens = self._estimate_tokens(response.text)
            
            content = response.text
            
            # Check if response contains tool call
            if tools and self._is_tool_call(content):
                tool_result = self._execute_tool_from_response(content, tools)
                
                # Generate final response with tool result
                final_prompt = f"{prompt}\n\nTool result: {json.dumps(tool_result)}"
                final_response = model.generate_content(final_prompt)
                
                content = final_response.text
                output_tokens += self._estimate_tokens(content)
            
            # Parse response based on pydantic_model
            if pydantic_model:
                response_obj = self._parse_pydantic_response(content, pydantic_model)
            else:
                response_obj = content
            
            return {
                'response': response_obj,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'model_used': model_name
            }
            
        except Exception as e:
            raise Exception(f"Google AI API error: {str(e)}")

    def _read_file_content(self, file_path: str) -> str:
        """Read file content as text."""
        if file_path.lower().endswith('.pdf'):
            # Use pdf_reader tool if available
            from src.tools import pdf_reader
            return pdf_reader(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _describe_tools(self, tools: Dict[str, Any]) -> str:
        """Generate a description of available tools."""
        descriptions = []
        for tool_name, tool_func in tools.items():
            import inspect
            doc = inspect.getdoc(tool_func) or f"Function {tool_name}"
            sig = inspect.signature(tool_func)
            params = []
            for param_name, param in sig.parameters.items():
                params.append(f"{param_name}: {param.annotation.__name__ if param.annotation != param.empty else 'Any'}")
            
            descriptions.append(f"- {tool_name}({', '.join(params)}): {doc}")
        
        return "\n".join(descriptions)

    def _is_tool_call(self, content: str) -> bool:
        """Check if the response contains a tool call."""
        try:
            data = json.loads(content)
            return 'tool_name' in data and 'tool_args' in data
        except:
            # Try to find JSON in the content
            import re
            json_match = re.search(r'\{.*"tool_name".*\}', content, re.DOTALL)
            return json_match is not None

    def _execute_tool_from_response(self, content: str, tools: Dict[str, Any]) -> Any:
        """Execute a tool based on the response content."""
        try:
            data = json.loads(content)
        except:
            # Try to extract JSON from the content
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse tool call from response")
        
        tool_name = data.get('tool_name')
        tool_args = data.get('tool_args', {})
        
        if tool_name not in tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return tools[tool_name](**tool_args)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on text length."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4

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
