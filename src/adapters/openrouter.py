from typing import Optional, Dict, Any
from src.adapters.base_adapter import BaseAdapter
from src.cost_tracker import CostTracker
import os

class OpenRouterAdapter(BaseAdapter):
    def __init__(self, model_name: str, cost_tracker: Optional[CostTracker] = None):
        super().__init__(model_name, cost_tracker) # model_name should be like "google/gemini-pro"
        # In a real scenario, initialize the OpenRouter client (likely OpenAI compatible)
        # from openai import OpenAI
        # self.client = OpenAI(
        #   base_url="https://openrouter.ai/api/v1",
        #   api_key=os.getenv("OPENROUTER_API_KEY"),
        # )
        print(f"OpenRouterAdapter initialized for model: {self.model_name}")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # This is a placeholder for actual OpenRouter API interaction.
        messages = input_data.get("messages", [])
        file_content_data = input_data.get("file_content")

        prompt_text = ""
        if messages and messages[0].get("role") == "user":
            prompt_text = messages[0].get("content", "")
            if isinstance(prompt_text, list): # Handle multimodal content
                prompt_text = next((item.get("text", "") for item in prompt_text if item.get("type") == "text"), "")

        mock_response_text = f"Mocked OpenRouter response for model '{self.model_name}' to prompt: '{str(prompt_text)[:50]}...'"
        if file_content_data:
            mock_response_text += f" with file '{file_content_data.get('filename')}'"
        # Cost can vary significantly on OpenRouter depending on the underlying model
        mock_cost_info = {"tokens_used": 105, "cost": 0.00018} # Example cost

        return {
            "content": {"text": mock_response_text, "tool_calls": []},
            "cost_info": mock_cost_info
        }
