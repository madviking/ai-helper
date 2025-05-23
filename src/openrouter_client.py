import requests
import time

class OpenRouterAPIError(Exception):
    """Custom exception for OpenRouter API errors."""
    pass

class OpenRouterClient:
    MODELS_URL = "https://openrouter.ai/api/v1/models"
    CACHE_DURATION_SECONDS = 24 * 60 * 60  # 24 hours

    def __init__(self):
        self._cached_models_data = None
        self._last_fetch_time = 0
        # self.requests_session = requests.Session() # Using requests.get directly for simplicity

    def _fetch_from_api(self):
        """
        Fetches model data from the OpenRouter API.
        """
        try:
            response = requests.get(self.MODELS_URL, timeout=10) # Add timeout
            response.raise_for_status() # Raises HTTPError for 4xx/5xx
            # It's good practice to check if 'data' key exists, though tests might assume it.
            json_response = response.json()
            if "data" not in json_response:
                raise OpenRouterAPIError("'data' key not found in API response.")
            return json_response["data"] # Assuming 'data' field contains the list of models
        except requests.exceptions.RequestException as e: # Catch HTTPError, ConnectionError, Timeout, etc.
            raise OpenRouterAPIError(f"API request failed: {e}")
        except json.JSONDecodeError as e: # Catch potential JSON parsing errors
             raise OpenRouterAPIError(f"Failed to decode JSON response: {e}")


    def get_models(self):
        """
        Retrieves models data, using cache if valid, otherwise fetches from API.
        """
        current_time = time.time()
        # 1. Check cache: current_time - self._last_fetch_time < CACHE_DURATION_SECONDS
        if self._cached_models_data is not None and \
           (current_time - self._last_fetch_time) < self.CACHE_DURATION_SECONDS:
            # 2. If cache valid and not empty, return self._cached_models_data
            return self._cached_models_data
        
        # 3. Else, call self._fetch_from_api()
        models_data = self._fetch_from_api()
        
        # 4. Store result in self._cached_models_data, update self._last_fetch_time = time.time()
        self._cached_models_data = models_data
        self._last_fetch_time = time.time() # Use current_time which is already fetched
        
        # 5. Return self._cached_models_data
        return self._cached_models_data
            
    def get_model_details(self, model_id: str):
        """
        Retrieves details for a specific model ID.
        """
        models = self.get_models() # This will ensure data is fetched/cache is used
        if models: # Check if models is not None and not empty
            for model in models:
                if model.get("id") == model_id:
                    return model
        return None # Or raise error if preferred by tests (tests expect None)

# Need to import json for the _fetch_from_api method if json.JSONDecodeError is caught explicitly
import json
