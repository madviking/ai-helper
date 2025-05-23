import unittest
import os
import base64
import json 
from unittest.mock import patch, MagicMock, ANY

from openai.types.chat import ChatCompletionMessage, ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from openai.types.completion_usage import CompletionUsage

from src.adapters.openai import OpenAIAdapter
from src.adapters.google import GoogleAdapter
from src.adapters.anthropic import AnthropicAdapter
from src.cost_tracker import CostTracker
from src.tools import calculator

import google.generativeai as genai # Reverted to standard import for Gemini SDK
from google.generativeai import types as genai_types
from anthropic import Anthropic, APIError # Import Anthropic for spec and APIError
from anthropic.types import Message, ToolUseBlock, TextBlock, Usage as AnthropicUsage

# Helper to create a mock OpenAI ChatCompletion object
def create_mock_chat_completion(text_content: str = None, tool_calls_data: list = None, usage_data: dict = None) -> ChatCompletion:
    tool_calls = None
    if tool_calls_data:
        tool_calls = []
        for tc_data in tool_calls_data:
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tc_data["id"],
                    type=tc_data["type"],
                    function=Function(name=tc_data["function"]["name"], arguments=tc_data["function"]["arguments"])
                )
            )
    message = ChatCompletionMessage(role="assistant", content=text_content, tool_calls=tool_calls)
    usage = None
    if usage_data:
        usage = CompletionUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
    return ChatCompletion(
        id="chatcmpl-mock",
        choices=[{"finish_reason": "stop" if not tool_calls else "tool_calls", "index": 0, "message": message}],
        created=1677652288, model="gpt-3.5-turbo", object="chat.completion", usage=usage
    )

class TestOpenAIAdapterIntegration(unittest.TestCase):
    def setUp(self):
        self.cost_tracker = CostTracker()
        self.getenv_patcher = patch.dict(os.environ, {"OPENAI_API_KEY": "fake_test_key"})
        self.mock_getenv = self.getenv_patcher.start()
        self.addCleanup(self.getenv_patcher.stop)
        self.adapter = OpenAIAdapter(model_name="gpt-3.5-turbo", cost_tracker=self.cost_tracker)

    @patch('openai.resources.chat.completions.Completions.create')
    def test_process_simple_prompt(self, mock_openai_create):
        mock_response_text = "Hello from mock OpenAI!"
        mock_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        mock_openai_create.return_value = create_mock_chat_completion(text_content=mock_response_text, usage_data=mock_usage)
        input_data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = self.adapter.process(input_data)
        mock_openai_create.assert_called_once()
        called_args = mock_openai_create.call_args[1]
        self.assertEqual(called_args["model"], "gpt-3.5-turbo")
        self.assertEqual(called_args["messages"][0]["content"], "Hello")
        self.assertEqual(result["content"]["text"], mock_response_text)
        self.assertFalse(result["content"]["tool_calls"])
        expected_cost = self.cost_tracker.calculate_cost("gpt-3.5-turbo", mock_usage["prompt_tokens"], mock_usage["completion_tokens"])
        self.assertAlmostEqual(result["cost_info"]["cost"], expected_cost, places=7)

    @patch('openai.resources.chat.completions.Completions.create')
    def test_process_with_tool_call_request_and_response(self, mock_openai_create):
        tool_call_id = "call_calc123"
        mock_tool_call_data = [{"id": tool_call_id, "type": "function", "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'}}]
        mock_usage1 = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        mock_openai_create.return_value = create_mock_chat_completion(tool_calls_data=mock_tool_call_data, usage_data=mock_usage1)
        calc_tool_schema = {"type": "function", "function": {"name": "calculator", "description": "Calculates stuff", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}
        input_data1 = {"messages": [{"role": "user", "content": "Calculate 2+2"}], "tools_details": [calc_tool_schema]}
        result1 = self.adapter.process(input_data1)
        self.assertIsNone(result1["content"]["text"])
        self.assertEqual(result1["content"]["tool_calls"][0]["function"]["arguments"], '{"expression":"2+2"}')
        
        mock_openai_create.reset_mock()
        final_text_response = "The result is 4."
        mock_usage2 = {"prompt_tokens": 40, "completion_tokens": 8, "total_tokens": 48}
        mock_openai_create.return_value = create_mock_chat_completion(text_content=final_text_response, usage_data=mock_usage2)
        input_data2 = {"messages": [{"role": "user", "content": "Calculate 2+2"}, {"role": "assistant", "tool_calls": result1["content"]["tool_calls"]}, {"role": "tool", "tool_call_id": tool_call_id, "name": "calculator", "content": "4"}], "tools_details": [calc_tool_schema]}
        result2 = self.adapter.process(input_data2)
        self.assertEqual(result2["content"]["text"], final_text_response)
        self.assertFalse(result2["content"]["tool_calls"])

    @patch('openai.resources.chat.completions.Completions.create')
    @patch('src.adapters.openai.get_mime_type')
    @patch('src.adapters.openai.base64.b64encode')
    def test_process_with_image_file(self, mock_b64encode, mock_get_mime_type, mock_openai_create):
        mock_get_mime_type.return_value = "image/png"
        mock_b64encode.return_value = b"fake_base64_encoded_image_data"
        mock_response_text = "Image received and described."
        mock_usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        mock_openai_create.return_value = create_mock_chat_completion(text_content=mock_response_text, usage_data=mock_usage)
        file_content_data = {"filename": "test_image.png", "content_bytes": b"dummy_image_bytes"}
        input_data = {"messages": [{"role": "user", "content": "Describe this image"}], "file_content": file_content_data}
        result = self.adapter.process(input_data)
        called_args = mock_openai_create.call_args[1]
        user_message_content = called_args["messages"][0]["content"]
        self.assertEqual(user_message_content[1]["image_url"]["url"], "data:image/png;base64,fake_base64_encoded_image_data")
        self.assertEqual(result["content"]["text"], mock_response_text)

# Helper to create a mock Gemini GenerateContentResponse object
def create_mock_gemini_response(text_content: str = None, function_calls_data: list = None, usage_metadata_data: dict = None) -> genai_types.GenerateContentResponse:
    parts = []
    if text_content:
        parts.append(genai_types.Part(text=text_content))
    if function_calls_data:
        for fc_data in function_calls_data: 
            parts.append(genai_types.Part(function_call=genai_types.FunctionCall(name=fc_data["name"], args=fc_data["args"])))
            
    mock_response = MagicMock(spec=genai_types.GenerateContentResponse)
    mock_response.parts = parts
    
    if usage_metadata_data:
        mock_response.usage_metadata = genai_types.UsageMetadata(
            prompt_token_count=usage_metadata_data.get("prompt_token_count",0),
            candidates_token_count=usage_metadata_data.get("candidates_token_count",0),
            total_token_count=usage_metadata_data.get("total_token_count",0)
        )
    else:
        mock_response.usage_metadata = MagicMock(spec=genai_types.UsageMetadata)
        mock_response.usage_metadata.prompt_token_count = 0
        mock_response.usage_metadata.candidates_token_count = 0
        mock_response.usage_metadata.total_token_count = 0
        
    mock_response.prompt_feedback = MagicMock()
    mock_response.prompt_feedback.block_reason = None
    return mock_response

class TestGoogleAdapterIntegration(unittest.TestCase):
    def setUp(self):
        self.cost_tracker = CostTracker()
        self.getenv_patcher = patch.dict(os.environ, {"GOOGLE_GEMINI_API_KEY": "fake_google_key"})
        self.mock_getenv = self.getenv_patcher.start()
        self.addCleanup(self.getenv_patcher.stop)
        
        # Mock the client instance that GoogleAdapter will create
        self.mock_gemini_client_instance = MagicMock(spec=genai.GenerativeModel)
        self.generative_model_patcher = patch('google.generativeai.GenerativeModel', return_value=self.mock_gemini_client_instance)
        self.mock_generative_model_constructor = self.generative_model_patcher.start()
        self.addCleanup(self.generative_model_patcher.stop)
        
        self.adapter = GoogleAdapter(model_name="gemini-1.5-flash", cost_tracker=self.cost_tracker)
        # self.adapter.client is now self.mock_gemini_client_instance
        # So, for tests, we set expectations on self.mock_gemini_client_instance.generate_content

    def test_process_simple_prompt_google(self):
        mock_response_text = "Hello from mock Gemini!"
        mock_usage = {"prompt_token_count": 12, "candidates_token_count": 8, "total_token_count": 20}
        # Set the return value on the instance that GoogleAdapter.client refers to
        self.adapter.client.generate_content.return_value = create_mock_gemini_response(text_content=mock_response_text, usage_metadata_data=mock_usage)
        
        input_data = {"messages": [{"role": "user", "content": "Hello Gemini"}]}
        result = self.adapter.process(input_data)
        
        self.adapter.client.generate_content.assert_called_once()
        called_args = self.adapter.client.generate_content.call_args[1]
        self.assertEqual(called_args["contents"][0]["parts"][0].text, "Hello Gemini")
        self.assertEqual(result["content"]["text"], mock_response_text)

    def test_process_with_tool_call_google(self):
        tool_name = "get_weather"; tool_args_dict = {"location": "London"}
        mock_fc_data = [{"name": tool_name, "args": tool_args_dict}]
        mock_usage1 = {"prompt_token_count": 25, "candidates_token_count": 15, "total_token_count": 40}
        self.adapter.client.generate_content.return_value = create_mock_gemini_response(function_calls_data=mock_fc_data, usage_metadata_data=mock_usage1)
        
        tool_schema = {"type": "function", "function": {"name": tool_name, "description": "Gets weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}
        input_data1 = {"messages": [{"role": "user", "content": "Weather in London?"}], "tools_details": [tool_schema]}
        result1 = self.adapter.process(input_data1)
        self.assertEqual(json.loads(result1["content"]["tool_calls"][0]["function"]["arguments"]), tool_args_dict)
        
        self.adapter.client.generate_content.reset_mock()
        final_text = "Weather is sunny."
        mock_usage2 = {"prompt_token_count": 50, "candidates_token_count": 10, "total_token_count": 60}
        self.adapter.client.generate_content.return_value = create_mock_gemini_response(text_content=final_text, usage_metadata_data=mock_usage2)
        
        input_data2 = {"messages": [{"role": "user", "content": "Weather in London?"}, {"role": "assistant", "tool_calls": result1["content"]["tool_calls"]}, {"role": "tool", "name": tool_name, "content": '{"temp": "15C"}'}], "tools_details": [tool_schema]}
        result2 = self.adapter.process(input_data2)
        self.assertEqual(result2["content"]["text"], final_text)

# Helper to create a mock Anthropic Message object
def create_mock_anthropic_message(content_blocks: list, usage_data: dict = None, stop_reason: str = "end_turn") -> Message:
    usage = None
    if usage_data:
        usage = AnthropicUsage(input_tokens=usage_data.get("input_tokens", 0), output_tokens=usage_data.get("output_tokens", 0))
    return Message(id="msg_mock_anthropic", content=content_blocks, model="claude-3", role="assistant", stop_reason=stop_reason, type="message", usage=usage)

class TestAnthropicAdapterIntegration(unittest.TestCase):
    def setUp(self):
        self.cost_tracker = CostTracker()
        self.getenv_patcher = patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_anthropic_key"})
        self.mock_getenv = self.getenv_patcher.start()
        self.addCleanup(self.getenv_patcher.stop)

        # Mock the client instance that AnthropicAdapter will create
        self.mock_anthropic_client_instance = MagicMock(spec=Anthropic)
        self.anthropic_constructor_patcher = patch('anthropic.Anthropic', return_value=self.mock_anthropic_client_instance)
        self.mock_anthropic_constructor = self.anthropic_constructor_patcher.start()
        self.addCleanup(self.anthropic_constructor_patcher.stop)
        
        self.adapter = AnthropicAdapter(model_name="claude-3-haiku-20240307", cost_tracker=self.cost_tracker)
        # self.adapter.client is now self.mock_anthropic_client_instance

    def test_process_simple_prompt_anthropic(self):
        mock_response_text = "Hello from mock Anthropic!"
        mock_usage = {"input_tokens": 10, "output_tokens": 7}
        # Set return_value on the messages.create method of the mocked client instance
        self.adapter.client.messages.create.return_value = create_mock_anthropic_message(
            content_blocks=[TextBlock(type="text", text=mock_response_text)], 
            usage_data=mock_usage
        )
        input_data = {"messages": [{"role": "user", "content": "Hello Anthropic"}]}
        result = self.adapter.process(input_data)
        
        self.adapter.client.messages.create.assert_called_once()
        called_args = self.adapter.client.messages.create.call_args[1]
        self.assertEqual(called_args["messages"][0]["content"][0]["text"], "Hello Anthropic")
        self.assertEqual(result["content"]["text"], mock_response_text)

    def test_process_with_tool_call_anthropic(self):
        tool_use_id = "tooluse_123abc"; tool_name = "get_user_info"; tool_input = {"user_id": "123"}
        mock_content_blocks1 = [ToolUseBlock(type="tool_use", id=tool_use_id, name=tool_name, input=tool_input)]
        mock_usage1 = {"input_tokens": 30, "output_tokens": 15}
        self.adapter.client.messages.create.return_value = create_mock_anthropic_message(
            content_blocks=mock_content_blocks1, usage_data=mock_usage1, stop_reason="tool_use"
        )
        tool_schema = {"type": "function", "function": {"name": tool_name, "description": "Gets user info", "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}}}
        input_data1 = {"messages": [{"role": "user", "content": "Info for user 123"}], "tools_details": [tool_schema]}
        result1 = self.adapter.process(input_data1)
        self.assertEqual(json.loads(result1["content"]["tool_calls"][0]["function"]["arguments"]), tool_input)

        self.adapter.client.messages.create.reset_mock()
        final_text = "User 123 is John Doe."
        mock_usage2 = {"input_tokens": 50, "output_tokens": 10}
        self.adapter.client.messages.create.return_value = create_mock_anthropic_message(
            content_blocks=[TextBlock(type="text", text=final_text)], usage_data=mock_usage2
        )
        input_data2 = {
            "messages": [
                {"role": "user", "content": "Info for user 123"}, 
                # AiHelper's _format_messages_for_anthropic needs to handle this assistant message correctly
                # For testing, we assume the adapter receives the history and formats it.
                # The crucial part is the 'tool' role message that follows.
                {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": tool_input}]},
                {"role": "tool", "tool_call_id": tool_use_id, "name": tool_name, "content": '{"name": "John Doe"}'}
            ], 
            "tools_details": [tool_schema]
        }
        result2 = self.adapter.process(input_data2)
        self.assertEqual(result2["content"]["text"], final_text)

class TestOtherIntegrationsPlaceholder(unittest.TestCase): 
    def test_openrouter_integration(self):
        pass

if __name__ == '__main__':
    unittest.main()
