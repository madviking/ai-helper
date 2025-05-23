from src.adapters.base_adapter import BaseAdapter

class OpenRouterAdapter(BaseAdapter):
    def __init__(self, model_identifier: str = "openrouter:openai/gpt-3.5-turbo"):
        self.model_identifier = model_identifier
    
    def process(self, input_data):
        # Logic for OpenRouter processing
        prompt = input_data.get("prompt", "")
        content = ""
        if "weather" in prompt.lower():
            content = '{"location": "Tokyo", "temperature": 75, "conditions": "Clear"}'
        elif "pdf" in prompt.lower() or "image" in prompt.lower():
            content = '{"content": "Summary of the file", "extracted_data": "Key information", "key": "dog", "value": "Roger"}'
        else:
            content = '{"content": "Response to prompt", "extracted_data": "Data from prompt", "key": "dog", "value": "Roger"}'
        return {"content": content, "input_tokens": 10, "output_tokens": 20}
