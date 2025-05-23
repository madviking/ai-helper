import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from pydantic import BaseModel
from typing import Optional

from src.adapters.base_adapter import BaseAdapter
from src.adapters.openai import OpenAIAdapter
from src.adapters.anthropic import AnthropicAdapter
from src.adapters.google import GoogleAdapter
from src.adapters.openrouter import OpenRouterAdapter

# Test Pydantic model
class TestModel(BaseModel):
    name: Optional[str] = None
    value: Optional[int] = None
    description: Optional[str] = None

class TestBaseAdapter(unittest.TestCase):
    def test_base_adapter_is_abstract(self):
        """Test that BaseAdapter cannot be instantiated"""
        with self.assertRaises(TypeError):
            BaseAdapter()
    
    def test_base_adapter_has_process_method(self):
        """Test that BaseAdapter defines process method"""
        self.assertTrue(hasattr(BaseAdapter, 'process'))

class TestOpenAIAdapter(unittest.TestCase):
    @patch('openai.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def setUp(self, mock_openai_class):
        """Set up test fixtures"""
        # Mock the OpenAI client
        self.mock_client = MagicMock()
        mock_openai_class.return_value = self.mock_client
        self.adapter = OpenAIAdapter()
    
    @patch('openai.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_openai_adapter_text_only(self, mock_openai_class):
        """Test OpenAI adapter with text-only input"""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"name": "test", "value": 42}'))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Process request
        input_data = {
            'prompt': 'Test prompt',
            'model': 'gpt-3.5-turbo',
            'pydantic_model': TestModel,
            'tools': None,
            'file': None
        }
        
        result = self.adapter.process(input_data)
        
        # Verify result
        self.assertIsInstance(result['response'], TestModel)
        self.assertEqual(result['response'].name, 'test')
        self.assertEqual(result['response'].value, 42)
        self.assertEqual(result['input_tokens'], 10)
        self.assertEqual(result['output_tokens'], 20)
        self.assertEqual(result['model_used'], 'gpt-3.5-turbo')
    
    @patch('openai.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_openai_adapter_with_tools(self, mock_openai_class):
        """Test OpenAI adapter with function calling"""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response with tool call
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = 'calculator'
        mock_tool_call.function.arguments = '{"expression": "2+2"}'
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[mock_tool_call]
            )
        )]
        mock_response.usage = MagicMock(prompt_tokens=15, completion_tokens=25)
        
        # First call returns tool request, second returns final response
        mock_client.chat.completions.create.side_effect = [
            mock_response,
            MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"name": "calc", "value": 4}'))],
                usage=MagicMock(prompt_tokens=20, completion_tokens=30)
            )
        ]
        
        # Mock tool
        mock_tool = MagicMock(return_value=4)
        
        # Process request
        input_data = {
            'prompt': 'Calculate 2+2',
            'model': 'gpt-3.5-turbo',
            'pydantic_model': TestModel,
            'tools': {'calculator': mock_tool},
            'file': None
        }
        
        result = self.adapter.process(input_data)
        
        # Verify tool was called
        mock_tool.assert_called_once_with(expression="2+2")
        
        # Verify result
        self.assertIsInstance(result['response'], TestModel)
        self.assertEqual(result['input_tokens'], 35)  # 15 + 20
        self.assertEqual(result['output_tokens'], 55)  # 25 + 30
    
    @patch('openai.OpenAI')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    @patch('builtins.open', new_callable=MagicMock)
    @patch('base64.b64encode')
    def test_openai_adapter_with_image(self, mock_b64encode, mock_open, mock_openai_class):
        """Test OpenAI adapter with image input"""
        # Mock file reading
        mock_open.return_value.__enter__.return_value.read.return_value = b'fake_image_data'
        mock_b64encode.return_value = b'encoded_image_data'
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"description": "An image"}'))]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=10)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Process request
        input_data = {
            'prompt': 'Describe this image',
            'model': 'gpt-4-vision-preview',
            'pydantic_model': TestModel,
            'tools': None,
            'file': 'test.png'
        }
        
        result = self.adapter.process(input_data)
        
        # Verify image was processed
        mock_open.assert_called_once_with('test.png', 'rb')
        
        # Verify result
        self.assertIsInstance(result['response'], TestModel)
        self.assertEqual(result['response'].description, 'An image')

class TestAnthropicAdapter(unittest.TestCase):
    @patch('anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    def setUp(self, mock_anthropic_class):
        """Set up test fixtures"""
        # Mock the Anthropic client
        self.mock_client = MagicMock()
        mock_anthropic_class.return_value = self.mock_client
        self.adapter = AnthropicAdapter()
    
    @patch('anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    def test_anthropic_adapter_text_only(self, mock_anthropic_class):
        """Test Anthropic adapter with text-only input"""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"name": "claude", "value": 100}')]
        mock_response.usage = MagicMock(input_tokens=15, output_tokens=25)
        mock_client.messages.create.return_value = mock_response
        
        # Process request
        input_data = {
            'prompt': 'Test prompt',
            'model': 'claude-3-opus-20240229',
            'pydantic_model': TestModel,
            'tools': None,
            'file': None
        }
        
        result = self.adapter.process(input_data)
        
        # Verify result
        self.assertIsInstance(result['response'], TestModel)
        self.assertEqual(result['response'].name, 'claude')
        self.assertEqual(result['response'].value, 100)
        self.assertEqual(result['input_tokens'], 15)
        self.assertEqual(result['output_tokens'], 25)
    
    @patch('anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'})
    def test_anthropic_adapter_with_tools(self, mock_anthropic_class):
        """Test Anthropic adapter with tool use"""
        # Mock Anthropic client
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client
        
        # Mock response with tool use
        mock_tool_use = MagicMock()
        mock_tool_use.type = 'tool_use'
        mock_tool_use.name = 'weather'
        mock_tool_use.input = {'location': 'Sofia'}
        mock_tool_use.id = 'tool_123'
        
        mock_response = MagicMock()
        mock_response.content = [mock_tool_use]
        mock_response.usage = MagicMock(input_tokens=20, output_tokens=30)
        
        # First call returns tool request, second returns final response
        mock_client.messages.create.side_effect = [
            mock_response,
            MagicMock(
                content=[MagicMock(text='{"name": "weather", "description": "Sunny in Sofia"}')],
                usage=MagicMock(input_tokens=25, output_tokens=35)
            )
        ]
        
        # Mock tool
        mock_tool = MagicMock(return_value={'temperature': 20, 'conditions': 'Sunny'})
        
        # Process request
        input_data = {
            'prompt': 'What is the weather in Sofia?',
            'model': 'claude-3-opus-20240229',
            'pydantic_model': TestModel,
            'tools': {'weather': mock_tool},
            'file': None
        }
        
        result = self.adapter.process(input_data)
        
        # Verify tool was called
        mock_tool.assert_called_once_with(location='Sofia')
        
        # Verify result
        self.assertIsInstance(result['response'], TestModel)
        self.assertEqual(result['input_tokens'], 45)  # 20 + 25
        self.assertEqual(result['output_tokens'], 65)  # 30 + 35

class TestGoogleAdapter(unittest.TestCase):
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = GoogleAdapter()
    
    @patch('google.genai.Client')
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
    def test_google_adapter_text_only(self, mock_client_class):
        """Test Google adapter with text-only input"""
        # Mock Google client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = '{"name": "gemini", "value": 50}'
        mock_response.usage_metadata = MagicMock(prompt_token_count=12, candidates_token_count=18)
        mock_client.models.generate_content.return_value = mock_response
        
        # Process request
        input_data = {
            'prompt': 'Test prompt',
            'model': 'gemini-2.5-flash-preview-05-20',
            'pydantic_model': TestModel,
            'tools': None,
            'file': None
        }
        
        result = self.adapter.process(input_data)
        
        # Verify result
        self.assertIsInstance(result['response'], TestModel)
        self.assertEqual(result['response'].name, 'gemini')
        self.assertEqual(result['response'].value, 50)
        self.assertEqual(result['input_tokens'], 12)
        self.assertEqual(result['output_tokens'], 18)
    
    @patch('google.genai.Client')
    @patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
    @patch('PIL.Image.open')
    def test_google_adapter_with_image(self, mock_image_open, mock_client_class):
        """Test Google adapter with image input"""
        # Mock image
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        
        # Mock Google client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = '{"description": "A beautiful sunset"}'
        mock_response.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=20)
        mock_client.models.generate_content.return_value = mock_response
        
        # Process request
        input_data = {
            'prompt': 'Describe this image',
            'model': 'gemini-2.5-flash-preview-05-20',
            'pydantic_model': TestModel,
            'tools': None,
            'file': 'sunset.jpg'
        }
        
        result = self.adapter.process(input_data)
        
        # Verify image was loaded
        mock_image_open.assert_called_once_with('sunset.jpg')
        
        # Verify result
        self.assertIsInstance(result['response'], TestModel)
        self.assertEqual(result['response'].description, 'A beautiful sunset')

class TestOpenRouterAdapter(unittest.TestCase):
    @patch.dict(os.environ, {'OPENROUTER_KEY': 'test_key'})
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = OpenRouterAdapter()
    
    @patch('openai.OpenAI')
    @patch.dict(os.environ, {'OPENROUTER_KEY': 'test_key'})
    def test_openrouter_adapter_model_mapping(self, mock_openai_class):
        """Test OpenRouter adapter with model name mapping"""
        # Mock OpenAI client (OpenRouter uses OpenAI-compatible API)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"name": "router", "value": 75}'))]
        mock_response.usage = MagicMock(prompt_tokens=8, completion_tokens=12)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Process request with various model formats
        test_cases = [
            ('openai/gpt-3.5-turbo', 'openai/gpt-3.5-turbo'),
            ('gpt-3.5-turbo', 'openai/gpt-3.5-turbo'),
            ('anthropic/claude-3', 'anthropic/claude-3'),
            ('claude-3', 'anthropic/claude-3'),
            ('google/gemini-2.5-flash-preview-05-20', 'google/gemini-2.5-flash-preview-05-20'),
            ('gemini-2.5-flash-preview-05-20', 'google/gemini-2.5-flash-preview-05-20')
        ]
        
        for input_model, expected_model in test_cases:
            input_data = {
                'prompt': 'Test prompt',
                'model': input_model,
                'pydantic_model': TestModel,
                'tools': None,
                'file': None
            }
            
            result = self.adapter.process(input_data)
            
            # Verify correct model was used
            call_args = mock_client.chat.completions.create.call_args
            self.assertEqual(call_args.kwargs['model'], expected_model)
            
            # Verify result
            self.assertIsInstance(result['response'], TestModel)
            self.assertEqual(result['model_used'], expected_model)
    
    @patch('openai.OpenAI')
    @patch.dict(os.environ, {'OPENROUTER_KEY': 'test_key'})
    def test_openrouter_adapter_headers(self, mock_openai_class):
        """Test OpenRouter adapter sets correct headers"""
        # Verify OpenRouter-specific configuration
        mock_openai_class.assert_called_with(
            api_key='test_key',
            base_url='https://openrouter.ai/api/v1',
            default_headers={
                'HTTP-Referer': 'https://github.com/ai-helper',
                'X-Title': 'AI Helper'
            }
        )

if __name__ == '__main__':
    unittest.main()
