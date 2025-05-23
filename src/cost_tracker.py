


class CostTracker:
    def __init__(self):
        self._total_cost = 0

        # maintained in three files (main level):
        # py_model_cost.json
        # llm_model_cost.json
        # total_cost.json

        self._cost_info = {}
        self._init_cost_info()

    def track_cost(self, cost_details):
        """Track the cost of a request based on provided details."""
        input_tokens = cost_details.get("input_tokens", 0)
        output_tokens = cost_details.get("output_tokens", 0)
        model = cost_details.get("model", "unknown_model")
        pydantic_model = cost_details.get("pydantic_model", "unknown_pydantic")
        
        self.add_cost(input_tokens, output_tokens, pydantic_model, model)

    def get_total_cost(self):
        return self._total_cost

    def reset_cost(self):
        self._total_cost = 0

    # Example method to add cost (could be based on tokens, etc.)
    def add_cost(self, input_tokens: int, output_tokens: int, pydantic_model: str, llm_model: str):
        """Calculate and add cost based on token usage for a specific model."""
        # Fetch cost per token from cached data if available
        input_cost_per_token = 0.000001  # Default fallback cost
        output_cost_per_token = 0.000002  # Default fallback cost
        
        for model_data in self._cost_info.get("model_data", []):
            if model_data.get("id") == llm_model:
                input_cost_per_token = model_data.get("pricing", {}).get("prompt", 0.000001) / 1000000  # Convert from per million to per token
                output_cost_per_token = model_data.get("pricing", {}).get("completion", 0.000002) / 1000000  # Convert from per million to per token
                break
        
        input_cost = input_tokens * input_cost_per_token
        output_cost = output_tokens * output_cost_per_token
        total_cost = input_cost + output_cost
        
        self._total_cost += total_cost
        
        # Update cost info for pydantic model
        if pydantic_model not in self._cost_info["pydantic_model_cost"]:
            self._cost_info["pydantic_model_cost"][pydantic_model] = 0
        self._cost_info["pydantic_model_cost"][pydantic_model] += total_cost
        
        # Update cost info for LLM model
        if llm_model not in self._cost_info["llm_model_cost"]:
            self._cost_info["llm_model_cost"][llm_model] = 0
        self._cost_info["llm_model_cost"][llm_model] += total_cost
        
        # Update total cost info
        self._cost_info["total_cost"]["total"] += total_cost

    """
    1) pull cost information from https://openrouter.ai/api/v1/models (no auth required)
    2) save and cache for 1 day. models.json
    """
    def _init_cost_info(self):
        """Initialize cost information structure and fetch from API if necessary."""
        import json
        import os
        import time
        import requests
        
        cache_file = "models.json"
        cache_duration = 86400  # 1 day in seconds
        
        # Check if cached data exists and is recent
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                cache_time = cache_data.get("timestamp", 0)
                if time.time() - cache_time < cache_duration:
                    self._cost_info = {
                        "pydantic_model_cost": {},
                        "llm_model_cost": {},
                        "total_cost": {"total": 0},
                        "model_data": cache_data.get("data", [])
                    }
                    return
        
        # Fetch data from OpenRouter API if no valid cache
        try:
            response = requests.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            model_data = response.json().get("data", [])
            
            # Save to cache with timestamp
            cache_data = {
                "timestamp": time.time(),
                "data": model_data
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self._cost_info = {
                "pydantic_model_cost": {},
                "llm_model_cost": {},
                "total_cost": {"total": 0},
                "model_data": model_data
            }
        except Exception as e:
            # Fallback to empty data if API fetch fails
            self._cost_info = {
                "pydantic_model_cost": {},
                "llm_model_cost": {},
                "total_cost": {"total": 0},
                "model_data": []
            }
            print(f"Failed to fetch cost data from OpenRouter API: {str(e)}")
