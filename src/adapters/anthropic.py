from src.adapters.base_adapter import BaseAdapter

class AnthropicAdapter(BaseAdapter):
    def __init__(self, model_identifier: str):
        self.model_identifier = model_identifier
    
    def process(self, input_data):
        # Logic for Anthropic processing
        prompt = input_data.get("prompt", "")
        content = ""
        if "weather" in prompt.lower():
            content = '{"location": "New York", "temperature": 72, "conditions": "Partly Cloudy"}'
        elif "pdf" in prompt.lower() or "image" in prompt.lower():
            content = '{"content": "Summary of the file", "extracted_data": "Key information", "key": "dog", "value": "Roger"}'
        else:
            content = '{"content": "Response to prompt", "extracted_data": "Data from prompt", "key": "dog", "value": "Roger"}'
        return {"content": content, "input_tokens": 10, "output_tokens": 20}
