import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


class CostTracker:
    def __init__(self):
        self._total_cost = 0.0
        self._cost_info = {}
        self._py_model_costs = {}
        self._llm_model_costs = {}
        
        # Load existing costs if available
        self._load_existing_costs()
        
        # Initialize cost information from API
        self._init_cost_info()

    def _load_existing_costs(self):
        """Load existing cost data from JSON files."""
        # Load total cost
        if os.path.exists('total_cost.json'):
            try:
                with open('total_cost.json', 'r') as f:
                    data = json.load(f)
                    self._total_cost = data.get('total', 0.0)
            except Exception:
                pass
        
        # Load py_model costs
        if os.path.exists('py_model_cost.json'):
            try:
                with open('py_model_cost.json', 'r') as f:
                    self._py_model_costs = json.load(f)
            except Exception:
                self._py_model_costs = {}
        
        # Load llm_model costs
        if os.path.exists('llm_model_cost.json'):
            try:
                with open('llm_model_cost.json', 'r') as f:
                    self._llm_model_costs = json.load(f)
            except Exception:
                self._llm_model_costs = {}

    def _init_cost_info(self):
        """
        Pull cost information from https://openrouter.ai/api/v1/models (no auth required)
        Save and cache for 1 day in models.json
        """
        models_file = 'models.json'
        
        # Check if cached file exists and is recent (less than 1 day old)
        if os.path.exists(models_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(models_file))
            if file_age < timedelta(days=1):
                # Load from cache
                try:
                    with open(models_file, 'r') as f:
                        data = json.load(f)
                        self._process_models_data(data)
                        return
                except Exception:
                    pass
        
        # Fetch from API
        try:
            response = requests.get('https://openrouter.ai/api/v1/models')
            if response.status_code == 200:
                data = response.json()
                
                # Save to cache
                with open(models_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                self._process_models_data(data)
        except Exception:
            # If API fails, continue with empty cost info
            pass

    def _process_models_data(self, data: Dict[str, Any]):
        """Process models data and extract pricing information."""
        if 'data' in data:
            for model in data['data']:
                model_id = model.get('id', '')
                pricing = model.get('pricing', {})
                
                if model_id and pricing:
                    self._cost_info[model_id] = {
                        'pricing': {
                            'prompt': pricing.get('prompt', '0'),
                            'completion': pricing.get('completion', '0')
                        }
                    }

    def track_cost(self, cost_details: Dict[str, Any]):
        """Track cost based on details (e.g., model, tokens, price)."""
        # Extract details
        model = cost_details.get('model')
        input_tokens = cost_details.get('input_tokens', 0)
        output_tokens = cost_details.get('output_tokens', 0)
        direct_cost = cost_details.get('cost')
        
        if direct_cost is not None:
            # If direct cost is provided, use it
            self._total_cost += direct_cost
        elif model and (input_tokens or output_tokens):
            # Calculate cost from tokens
            self.add_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pydantic_model=cost_details.get('pydantic_model'),
                llm_model=model
            )

    def get_total_cost(self) -> float:
        """Get the total accumulated cost."""
        return self._total_cost

    def reset_cost(self):
        """Reset all costs to zero."""
        self._total_cost = 0.0
        self._py_model_costs = {}
        self._llm_model_costs = {}
        
        # Update files
        self._save_costs()

    def add_cost(self, input_tokens: int, output_tokens: int, pydantic_model: Optional[str], llm_model: str):
        """
        Calculate and add cost using input and output tokens.
        
        Args:
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            pydantic_model: Name of the Pydantic model used (if any)
            llm_model: Name of the LLM model used
        """
        # Look up pricing for the model
        model_info = self._cost_info.get(llm_model)
        
        if not model_info:
            # Try without provider prefix for OpenAI models
            if llm_model.startswith('gpt-'):
                model_info = self._cost_info.get(f'openai/{llm_model}')
            
            if not model_info:
                # Model not found, can't calculate cost
                return
        
        # Get pricing (prices are per 1M tokens in OpenRouter API)
        prompt_price = float(model_info['pricing']['prompt'])
        completion_price = float(model_info['pricing']['completion'])
        
        # Calculate cost (prices are per 1K tokens in OpenRouter API)
        input_cost = (input_tokens * prompt_price) / 1_000
        output_cost = (output_tokens * completion_price) / 1_000
        total_cost = input_cost + output_cost
        
        # Update total cost
        self._total_cost += total_cost
        
        # Update per-model costs
        if pydantic_model:
            if pydantic_model not in self._py_model_costs:
                self._py_model_costs[pydantic_model] = 0.0
            self._py_model_costs[pydantic_model] += total_cost
        
        if llm_model not in self._llm_model_costs:
            self._llm_model_costs[llm_model] = 0.0
        self._llm_model_costs[llm_model] += total_cost
        
        # Save updated costs
        self._save_costs()

    def _save_costs(self):
        """Save cost data to JSON files."""
        # Save total cost
        with open('total_cost.json', 'w') as f:
            json.dump({
                'total': self._total_cost,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save py_model costs
        with open('py_model_cost.json', 'w') as f:
            json.dump(self._py_model_costs, f, indent=2)
        
        # Save llm_model costs
        with open('llm_model_cost.json', 'w') as f:
            json.dump(self._llm_model_costs, f, indent=2)
