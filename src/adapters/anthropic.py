import os
from src.adapters.base_adapter import BaseAdapter
from anthropic import Anthropic # Import the actual Anthropic client

class AnthropicAdapter(BaseAdapter):
    def __init__(self):
        # Initialize the Anthropic client with the API key from environment variables
        # TODO: Handle missing API key more gracefully
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        # TODO: Allow model to be configured

    def process(self, input_data):
        print(f"Processing data with AnthropicAdapter: {input_data}")

        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Prompt is missing in input_data"}

        # Prepare messages for the API call
        messages = [{"role": "user", "content": prompt}]

        # TODO: Handle other input_data parameters like tools, model, etc.
        try:
            # Make the actual API call
            response = self.client.messages.create(
                model="claude-3-opus-20240229", # Example model, should be configurable
                max_tokens=1024, # Example max_tokens, should be configurable
                messages=messages
            )

            # Extract relevant information from the response
            # Assuming the response structure from the Anthropic library
            model_output = ""
            if response.content:
                # Concatenate text content blocks
                model_output = "".join([block.text for block in response.content if block.type == 'text'])

            usage = getattr(response, 'usage', None)
            output_tokens = getattr(usage, 'output_tokens', 0)
            # TODO: Calculate actual cost based on tokens and model

            cost_details = {
                "cost": 0.0, # Placeholder cost
                "output_tokens": output_tokens,
                "model": response.model # Use the model name from the response
            }


            return {
                "model_output": model_output,
                "cost_details": cost_details
            }

        except Exception as e:
            return {"error": f"Error during Anthropic API call: {e}"}
