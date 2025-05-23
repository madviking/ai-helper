import openai
from src.adapters.base_adapter import BaseAdapter

class OpenAIAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str):
        """
        Initializes the OpenAIAdapter.

        Args:
            api_key (str): The API key for OpenAI.
            model (str): The model name to use (e.g., "gpt-3.5-turbo").
        """
        self.api_key = api_key
        self.model = model

    def process(self, input_data: str) -> str:
        """
        Processes the input string using the OpenAI ChatCompletion API.

        Args:
            input_data (str): The input string to send to the AI.

        Returns:
            str: The content of the AI's response.
        """
        messages = [{"role": "user", "content": input_data}]
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            api_key=self.api_key
        )
        
        return response.choices[0].message.content
