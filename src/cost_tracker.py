import logging

# Configure basic logging
# Use INFO for more verbose logging if needed during dev, WARNING for production
logging.basicConfig(level=logging.WARNING) 
logger = logging.getLogger(__name__)

class CostTracker:
    def __init__(self, openrouter_client=None):
        self._total_cost = 0.0 # Ensure total_cost is a float
        self.openrouter_client = openrouter_client

    def track_cost(self, cost_details):
        current_transaction_cost = 0.0

        if isinstance(cost_details, (int, float)):
            current_transaction_cost = float(cost_details)
        elif isinstance(cost_details, dict):
            # Essential keys for any token-based calculation
            # Use .get() for safety, defaulting to None if key is missing
            prompt_tokens_val = cost_details.get('prompt_tokens')
            completion_tokens_val = cost_details.get('completion_tokens')
            model_identifier_val = cost_details.get('model_identifier')

            # Strategy 1: Direct Pricing (highest priority if all direct pricing keys are present)
            cost_per_prompt_token_val = cost_details.get("cost_per_prompt_token")
            cost_per_completion_token_val = cost_details.get("cost_per_completion_token")

            if cost_per_prompt_token_val is not None and cost_per_completion_token_val is not None:
                if model_identifier_val is None: # model_identifier is logically part of a complete record
                    logger.warning("Direct pricing specified but 'model_identifier' missing. Adding 0 cost for this transaction.")
                elif prompt_tokens_val is None or completion_tokens_val is None:
                    # Ensure prompt_tokens_val and completion_tokens_val are not None before float conversion
                    # If one is None, this path implies an incomplete record for direct pricing.
                    logger.warning(f"Direct pricing for '{model_identifier_val}' but prompt/completion tokens missing or None. Adding 0 cost for this transaction.")
                else:
                    try:
                        current_transaction_cost = (float(prompt_tokens_val) * float(cost_per_prompt_token_val)) + \
                                                 (float(completion_tokens_val) * float(cost_per_completion_token_val))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error converting direct pricing/token values for '{model_identifier_val}': {e}. Adding 0 cost for this transaction.")
                        current_transaction_cost = 0.0 # Ensure cost remains 0 on error
            
            # Strategy 2: OpenRouter Pricing (if no complete direct pricing and model_identifier indicates OpenRouter)
            elif model_identifier_val is not None and isinstance(model_identifier_val, str) and \
                 model_identifier_val.startswith("openrouter:"):
                if not self.openrouter_client:
                    logger.warning(f"OpenRouter model '{model_identifier_val}' specified but no OpenRouter client provided. Adding 0 cost for this transaction.")
                elif prompt_tokens_val is None or completion_tokens_val is None: # Both token counts are essential
                    logger.warning(f"OpenRouter model '{model_identifier_val}' specified but prompt/completion tokens missing or None. Adding 0 cost for this transaction.")
                else:
                    actual_model_id = model_identifier_val.split("openrouter:", 1)[1]
                    if not actual_model_id: # Handle case like "openrouter:"
                         logger.warning(f"Invalid OpenRouter model identifier format: '{model_identifier_val}'. Adding 0 cost.")
                    else:
                        model_info = self.openrouter_client.get_model_details(actual_model_id)
                        
                        if model_info and isinstance(model_info.get("pricing"), dict):
                            pricing = model_info["pricing"]
                            prompt_price_str = pricing.get("prompt")
                            completion_price_str = pricing.get("completion")
                            
                            if prompt_price_str is not None and completion_price_str is not None:
                                try:
                                    current_transaction_cost = (float(prompt_tokens_val) * float(prompt_price_str)) + \
                                                             (float(completion_tokens_val) * float(completion_price_str))
                                except (ValueError, TypeError) as e:
                                    logger.warning(f"Error converting OpenRouter pricing/token values for '{actual_model_id}': {e}. Adding 0 cost for this transaction.")
                                    current_transaction_cost = 0.0 # Ensure cost remains 0 on error
                            else:
                                logger.warning(f"Incomplete pricing info (prompt/completion price missing) for OpenRouter model '{actual_model_id}'. Adding 0 cost for this transaction.")
                        else:
                            logger.warning(f"Could not retrieve valid pricing (pricing object or model info missing) for OpenRouter model '{actual_model_id}'. Adding 0 cost for this transaction.")
            # else: If it's a dictionary but doesn't fit the above (e.g., no direct pricing, not openrouter, or missing model_identifier for OR path)
            # then current_transaction_cost remains 0.0 as initialized. This covers:
            # - Simple dicts like {"model": "gpt-4", "tokens": 100}
            # - Dicts with model_identifier not starting with "openrouter:" and no direct pricing keys
            # - Dicts with model_identifier but missing other crucial fields for any strategy
            # Logger warnings for these specific fall-throughs can be added if more verbosity is desired.
            # The tests (e.g. test_track_cost_non_numeric_simple_dict_no_pricing_info, 
            # test_track_cost_non_openrouter_model_no_direct_pricing) expect 0 for these.

        # else: If cost_details is not an int, float, or dict (e.g., a list or string),
        # current_transaction_cost remains 0.0 as initialized.
        # Test 'test_track_cost_other_non_numeric_types' expects 0 for these.
        
        self._total_cost += current_transaction_cost

    def get_total_cost(self):
        return self._total_cost

    def reset_cost(self):
        self._total_cost = 0.0 # Ensure reset to float, consistent with initialization
