import unittest
from unittest.mock import patch, MagicMock
import os
from pydantic import BaseModel
from typing import Optional

from src.ai_helper import AIHelper
from py_models.weather_model import WeatherModel
from py_models.general_example_model import GeneralExampleModel

class TestIntegrations(unittest.TestCase):
    """Integration tests for the AI Helper system"""
    
    @patch('src.adapters.openai.OpenAI')
    @patch('src.adapters.anthropic.Anthropic')
    @patch('google.genai.Client')
    def test_multiple_providers_same_result(self, mock_google_client, mock_anthropic_client, mock_openai_client):
        """Test that different providers give similar results for the same prompt"""
        # Set up mock responses for each provider
        test_response = GeneralExampleModel(
            content="Test content",
            key="dog",
            value="Roger"
        )
        
        # Mock OpenAI response
        mock_openai = MagicMock()
        mock_openai_client.return_value = mock_openai
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=test_response.model_dump_json()))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20)
        )
        
        # Mock Anthropic response
        mock_anthropic = MagicMock()
        mock_anthropic_client.return_value = mock_anthropic
        mock_anthropic.messages.create.return_value = MagicMock(
            content=[MagicMock(text=test_response.model_dump_json())],
            usage=MagicMock(input_tokens=10, output_tokens=20)
        )
        
        # Mock Google response
        mock_google = MagicMock()
        mock_google_client.return_value = mock_google
        mock_google.models.generate_content.return_value = MagicMock(
            text=test_response.model_dump_json(),
            usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=20)
        )
        
        # Test models
        models_to_test = [
            'openai:gpt-3.5-turbo',
            'anthropic:claude-3',
            'google:gemini-2.5-flash-preview-05-20'
        ]
        
        results = []
        for model in models_to_test:
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test_key',
                'ANTHROPIC_API_KEY': 'test_key',
                'GOOGLE_API_KEY': 'test_key'
            }):
                helper = AIHelper(model)
                result = helper.ask(
                    "This is the test file we use. Key is 'dog' and value for that is 'Roger'",
                    model=GeneralExampleModel
                )
                results.append(result)
        
        # Verify all results are similar
        for result in results:
            self.assertIsInstance(result, GeneralExampleModel)
            self.assertEqual(result.key, "dog")
            self.assertEqual(result.value, "Roger")
    
    @patch('src.adapters.openai.OpenAI')
    @patch('src.adapters.openrouter.OpenAI')
    def test_openrouter_model_mapping(self, mock_openrouter_client, mock_openai_client):
        """Test that OpenRouter correctly maps model names"""
        # Mock responses
        mock_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"name": "test"}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20)
        )
        
        # Set up mocks
        mock_openai = MagicMock()
        mock_openai_client.return_value = mock_openai
        mock_openai.chat.completions.create.return_value = mock_response
        
        mock_openrouter = MagicMock()
        mock_openrouter_client.return_value = mock_openrouter
        mock_openrouter.chat.completions.create.return_value = mock_response
        
        # Test direct provider vs OpenRouter
        test_cases = [
            ('openai:gpt-3.5-turbo', 'openrouter:openai/gpt-3.5-turbo'),
            ('anthropic:claude-3', 'openrouter:anthropic/claude-3'),
            ('google:gemini-2.5-flash-preview-05-20', 'openrouter:google/gemini-2.5-flash-preview-05-20')
        ]
        
        for direct_model, openrouter_model in test_cases:
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test_key',
                'OPENROUTER_KEY': 'test_key'
            }):
                # Test direct provider
                helper1 = AIHelper(direct_model)
                result1 = helper1.ask("Test", model=GeneralExampleModel)
                
                # Test via OpenRouter
                helper2 = AIHelper(openrouter_model)
                result2 = helper2.ask("Test", model=GeneralExampleModel)
                
                # Both should work
                self.assertIsInstance(result1, GeneralExampleModel)
                self.assertIsInstance(result2, GeneralExampleModel)
    
    @patch('src.adapters.openai.OpenAI')
    @patch('src.tools.weather')
    def test_tool_integration(self, mock_weather_tool, mock_openai_client):
        """Test integration of tools with LLM"""
        # Mock weather tool
        mock_weather_tool.return_value = {
            'location': 'Sofia, Bulgaria',
            'temperature': 20,
            'conditions': 'Sunny'
        }
        
        # Mock OpenAI to request tool use
        mock_openai = MagicMock()
        mock_openai_client.return_value = mock_openai
        
        # First response requests tool
        tool_response = MagicMock()
        tool_response.choices = [MagicMock(
            message=MagicMock(
                content=None,
                tool_calls=[MagicMock(
                    function=MagicMock(
                        name='weather',
                        arguments='{"location": "Sofia"}'
                    )
                )]
            )
        )]
        tool_response.usage = MagicMock(prompt_tokens=15, completion_tokens=25)
        
        # Second response with final answer
        final_response = MagicMock()
        final_response.choices = [MagicMock(
            message=MagicMock(
                content='{"location": "Sofia, Bulgaria", "temperature": 20, "conditions": "Sunny"}'
            )
        )]
        final_response.usage = MagicMock(prompt_tokens=20, completion_tokens=30)
        
        mock_openai.chat.completions.create.side_effect = [tool_response, final_response]
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            helper = AIHelper('openai:gpt-3.5-turbo')
            helper.add_tool("weather", "Get weather information")
            
            result = helper.ask(
                "What is the weather like in Sofia?",
                tools=["weather"],
                model=WeatherModel
            )
            
            # Verify result
            self.assertIsInstance(result, WeatherModel)
            self.assertEqual(result.location, "Sofia, Bulgaria")
            self.assertEqual(result.temperature, 20)
            self.assertEqual(result.conditions, "Sunny")
            
            # Verify tool was called
            mock_weather_tool.assert_called_once_with(location="Sofia")
    
    @patch('src.adapters.openai.OpenAI')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('PyPDF2.PdfReader')
    def test_file_handling_pdf(self, mock_pdf_reader, mock_open, mock_openai_client):
        """Test PDF file handling integration"""
        # Mock PDF reading
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="PDF content page 1"))]
        mock_pdf_reader.return_value = mock_pdf
        
        # Mock OpenAI response
        mock_openai = MagicMock()
        mock_openai_client.return_value = mock_openai
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content='{"content": "Summary of PDF", "extracted_data": {"pages": 1}}'
            ))],
            usage=MagicMock(prompt_tokens=50, completion_tokens=30)
        )
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            helper = AIHelper('openai:gpt-3.5-turbo')
            result = helper.ask(
                "Please read this PDF and summarize it.",
                model=GeneralExampleModel,
                file="test.pdf"
            )
            
            # Verify result
            self.assertIsInstance(result, GeneralExampleModel)
            self.assertEqual(result.content, "Summary of PDF")
            self.assertIsNotNone(result.extracted_data)
    
    @patch('src.adapters.google.Client')
    @patch('PIL.Image.open')
    def test_file_handling_image(self, mock_image_open, mock_google_client):
        """Test image file handling integration"""
        # Mock image
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        
        # Mock Google response
        mock_google = MagicMock()
        mock_google_client.return_value = mock_google
        mock_google.models.generate_content.return_value = MagicMock(
            text='{"content": "This is an image of a sunset", "extracted_data": {"type": "sunset"}}',
            usage_metadata=MagicMock(prompt_token_count=100, candidates_token_count=40)
        )
        
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            helper = AIHelper('google:gemini-2.5-flash-preview-05-20')
            result = helper.ask(
                "Please describe this image.",
                model=GeneralExampleModel,
                file="sunset.png"
            )
            
            # Verify result
            self.assertIsInstance(result, GeneralExampleModel)
            self.assertIn("sunset", result.content.lower())
    
    def test_anthropic_no_file_support(self):
        """Test that Anthropic models properly handle lack of file support"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key'}):
            helper = AIHelper('anthropic:claude-3')
            
            # Should raise appropriate error for file input
            with self.assertRaises(ValueError) as context:
                helper.ask(
                    "Read this file",
                    model=GeneralExampleModel,
                    file="test.pdf"
                )
            
            self.assertIn("does not support file", str(context.exception))
    
    @patch('src.cost_tracker.CostTracker')
    @patch('src.adapters.openai.OpenAI')
    def test_cost_tracking_integration(self, mock_openai_client, mock_cost_tracker_class):
        """Test that costs are properly tracked across requests"""
        # Mock cost tracker
        mock_cost_tracker = MagicMock()
        mock_cost_tracker_class.return_value = mock_cost_tracker
        
        # Mock OpenAI
        mock_openai = MagicMock()
        mock_openai_client.return_value = mock_openai
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"name": "test"}'))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20)
        )
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Create helper with explicit cost tracker
            helper = AIHelper('openai:gpt-3.5-turbo', mock_cost_tracker)
            
            # Make multiple requests
            for i in range(3):
                helper.ask(f"Test {i}", model=GeneralExampleModel)
            
            # Verify cost tracking
            self.assertEqual(mock_cost_tracker.add_cost.call_count, 3)
            
            # Verify correct parameters
            for call in mock_cost_tracker.add_cost.call_args_list:
                args, kwargs = call
                self.assertEqual(kwargs['input_tokens'], 10)
                self.assertEqual(kwargs['output_tokens'], 20)
                self.assertEqual(kwargs['pydantic_model'], 'GeneralExampleModel')
                self.assertEqual(kwargs['llm_model'], 'gpt-3.5-turbo')
    
    def test_validation_and_fill_percentage(self):
        """Test that partial validation works and fill percentage is calculated"""
        # This would be tested with actual API calls in real integration tests
        # For now, we'll mock the behavior
        pass

if __name__ == '__main__':
    unittest.main()
