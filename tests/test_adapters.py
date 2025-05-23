import unittest
from unittest.mock import patch, MagicMock
import os
from pydantic import BaseModel
from openai import OpenAI # Ensure OpenAI is imported for type hinting if needed by mocks

from src.adapters.openai import OpenAIAdapter
from src.cost_tracker import CostTracker

# Define a simple Pydantic model for testing
class TestOutputModel(BaseModel):
    name: str
    age: int
    city: Optional[str] = None

class TestAdapters(unittest.TestCase):

    def setUp(self):
        # Set a dummy API key for tests if the adapter initialization requires it
        os.environ["OPENAI_API_KEY"] = "test_api_key"
        self.cost_tracker = CostTracker() # Or mock it if its interactions are complex

    def tearDown(self):
        # Clean up environment variables
        del os.environ["OPENAI_API_KEY"]

    @patch('src.adapters.openai.OpenAI') # Patch the OpenAI client in the adapter's module
    def test_openai_adapter_process_with_pydantic_ai(self, MockOpenAI):
        # Mock the OpenAI client instance and its methods
        mock_client_instance = MockOpenAI.return_value
        mock_completion = MagicMock()
        
        # This is what pydantic-ai's OpenAI instructor would typically handle.
        # For the adapter test, we need to simulate the adapter *using* pydantic-ai.
        # So, the adapter's 'process' method will internally use pydantic-ai's instructor.
        # The mock here should simulate the direct OpenAI API call if pydantic-ai
        # was NOT yet integrated, or if we are testing the adapter's direct interaction
        # before pydantic-ai is added to its internals.
        #
        # If pydantic-ai is integrated *within* the adapter, the adapter's 'process'
        # method will change. It will take a Pydantic model class and use
        # pydantic_ai.OpenAIInstructor.create(client=..., model=...)
        #
        # For now, let's assume we are testing the *new* behavior where the adapter
        # itself uses pydantic-ai. So, the adapter's 'process' method will be called
        # with a Pydantic model, and it should return an instance of that model.

        # The adapter will instantiate its own OpenAI client.
        # PydanticAI's OpenAIInstructor will then use this client.
        # The `create` method of the instructor is what we'd expect to be called.
        # So, we need to mock the behavior of `pydantic_ai.OpenAIInstructor(...).create(...)`
        # or more simply, mock the part of the adapter that *would* call the LLM
        # and ensure it returns a Pydantic model instance as if pydantic-ai did its job.

        # Let's refine the test to reflect that the adapter's `process` method
        # will be modified to use PydanticAI.
        # The adapter will receive the Pydantic model class.

        adapter = OpenAIAdapter(model_name="gpt-3.5-turbo", cost_tracker=self.cost_tracker)

        # We need to mock the part of the adapter that calls the LLM via PydanticAI.
        # PydanticAI's `OpenAIInstructor` would call `client.chat.completions.create`.
        # The adapter's `process` method will be updated to use `OpenAIInstructor`.
        # So, we mock `client.chat.completions.create` which `OpenAIInstructor` uses.

        # Simulate the LLM responding with data that can be parsed into TestOutputModel
        # PydanticAI expects the LLM to return a JSON string corresponding to the model.
        mock_llm_response_content = '{"name": "Cline", "age": 30, "city": "Testville"}'
        
        mock_chat_completion_message = MagicMock()
        mock_chat_completion_message.content = mock_llm_response_content
        
        mock_completion_choice = MagicMock()
        mock_completion_choice.message = mock_chat_completion_message
        
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [mock_completion_choice]
        mock_openai_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        # If the adapter uses pydantic_ai.OpenAIInstructor, it will call client.chat.completions.create
        # The instructor then parses the response.
        # So, the mock for `create` should return something that, when processed by PydanticAI's
        # parsing logic (which happens inside the instructor), yields the model instance.
        # PydanticAI's instructor itself makes the call and then parses.
        #
        # Let's assume the adapter's `process` method is updated like this (pseudo-code):
        # def process(self, ..., pydantic_model_class):
        #   from pydantic_ai import OpenAIInstructor
        #   instructor = OpenAIInstructor(client=self.client, model=pydantic_model_class)
        #   model_instance = instructor.create(messages=...)
        #   return {"content": {"model_instance": model_instance}, "cost_info": ...}

        # So, we mock `self.client.chat.completions.create` which `instructor.create` calls.
        mock_client_instance.chat.completions.create.return_value = mock_openai_response

        input_data = {
            "messages": [{"role": "user", "content": "Extract name and age."}],
            "pydantic_model_class": TestOutputModel # Pass the class itself
        }

        result = adapter.process(input_data)

        # Assertions
        self.assertIn("content", result)
        self.assertIn("model_instance", result["content"])
        model_instance = result["content"]["model_instance"]

        self.assertIsInstance(model_instance, TestOutputModel)
        self.assertEqual(model_instance.name, "Cline")
        self.assertEqual(model_instance.age, 30)
        self.assertEqual(model_instance.city, "Testville")

        mock_client_instance.chat.completions.create.assert_called_once()
        call_args = mock_client_instance.chat.completions.create.call_args
        
        # PydanticAI's instructor adds specific instructions to the messages
        # to guide the LLM to output JSON. We should check for that.
        # The exact prompt engineering might vary with pydantic-ai versions.
        # For now, let's assume the messages passed to OpenAI API are modified.
        # This part is tricky to assert without knowing pydantic-ai's exact internal prompt.
        # A simpler check is that the original user message is part of the call.
        self.assertEqual(call_args.kwargs['messages'][-1]['role'], 'user') # Or however pydantic-ai structures it
        self.assertIn("Extract name and age.", call_args.kwargs['messages'][-1]['content'])
        
        # Check if the model name was passed correctly
        self.assertEqual(call_args.kwargs['model'], "gpt-3.5-turbo")

        # Check cost tracking (simplified)
        self.assertIsNotNone(result.get("cost_info"))
        self.assertEqual(result["cost_info"]["prompt_tokens"], 10)
        self.assertEqual(result["cost_info"]["completion_tokens"], 20)


    def test_anthropic_adapter_process(self):
        # TODO: Implement similar test for Anthropic when PydanticAI support is clear/added
        pass

    def test_google_adapter_process(self):
        # TODO: Implement similar test for Google when PydanticAI support is clear/added
        pass

    def test_openrouter_adapter_process(self):
        # TODO: Implement similar test for OpenRouter when PydanticAI support is clear/added
        pass

if __name__ == '__main__':
    unittest.main()
