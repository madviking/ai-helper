
import json
import os
import time
from typing import Dict, Optional, Any

# Example pricing per 1M tokens (input, output)
# These are illustrative and might not be current.
# For OpenAI models, pricing is often per 1K tokens.
# GPT-3.5-turbo-0125: $0.50 / 1M input, $1.50 / 1M output
# GPT-4-turbo-preview (e.g., gpt-4-0125-preview): $10.00 / 1M input, $30.00 / 1M output
# For simplicity, storing per 1 token.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "default": {"input": 20 / 1_000_000, "output": 20 / 1_000_000}, # A generic default
    "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "gpt-3.5-turbo-0125": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
    "gpt-4-turbo-preview": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    "gpt-4-0125-preview": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    "gpt-4o": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000},
    # Add other models as needed, e.g., from OpenRouter, Anthropic, Google
}

class CostTracker:
    def __init__(self, pricing_data: Optional[Dict[str, Dict[str, float]]] = None):
        self._total_cost: float = 0.0
        self.model_pricing = pricing_data if pricing_data is not None else MODEL_PRICING
        
        # The idea of loading from OpenRouter is good for dynamic pricing
        # self._cost_info = {} # This could store fetched prices
        # self._init_cost_info() # Call to fetch/load prices

    def calculate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculates cost for a given model and token counts."""
        # Try to find specific model pricing, fallback to a generic version if model_name includes provider
        # e.g. "openai:gpt-3.5-turbo" -> "gpt-3.5-turbo"
        normalized_model_name = model_name.split(':')[-1].split('/')[-1]

        pricing = self.model_pricing.get(normalized_model_name)
        if not pricing: # Fallback for models like "openai/gpt-3.5-turbo" from openrouter
            pricing = self.model_pricing.get(model_name) 
            if not pricing:
                print(f"Warning: Pricing not found for model '{model_name}' (normalized: '{normalized_model_name}'). Using default if available, or cannot calculate cost.")
                pricing = self.model_pricing.get("default")
                if not pricing:
                    return None # Cannot calculate if no default and no specific match

        input_cost = prompt_tokens * pricing.get("input", 0)
        output_cost = completion_tokens * pricing.get("output", 0)
        return input_cost + output_cost

    def track_cost(self, cost_details: Dict[str, Any]):
        """
        Tracks cost. Expects cost_details to contain 'cost' if pre-calculated,
        or 'model_name', 'prompt_tokens', 'completion_tokens' to calculate it.
        """
        cost_to_add = 0.0
        if "cost" in cost_details and cost_details["cost"] is not None:
            cost_to_add = cost_details["cost"]
        elif all(k in cost_details for k in ("model_name", "prompt_tokens", "completion_tokens")):
            calculated = self.calculate_cost(
                cost_details["model_name"],
                cost_details["prompt_tokens"],
                cost_details["completion_tokens"]
            )
            if calculated is not None:
                cost_to_add = calculated
            else: # Could not calculate, so don't add to total_cost
                print(f"Could not calculate cost for: {cost_details}")
                return # Do not add to _total_cost if calculation failed
        else:
            # Not enough details to calculate or use pre-calculated cost
            print(f"Warning: Insufficient details to track cost: {cost_details}")
            return

        self._total_cost += cost_to_add
        # print(f"Cost added: {cost_to_add:.6f}, Total cost: {self._total_cost:.6f}")


    def get_total_cost(self) -> float:
        return self._total_cost

    def reset_cost(self):
        self._total_cost = 0.0
        # print("Cost tracker reset.")

    # _init_cost_info and add_cost (old version) can be re-evaluated later if needed
    # For now, focusing on direct calculation and tracking.
    def _init_cost_info(self):
        """
        Placeholder for initializing cost info, e.g., by fetching from OpenRouter.
        1) pull cost information from https://openrouter.ai/api/v1/models (no auth required)
        2) save and cache for 1 day. models.json
        """
        # Example:
        # self.model_pricing = self.load_pricing_from_file_or_api()
        pass
