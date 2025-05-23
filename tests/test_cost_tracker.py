import unittest
from unittest.mock import MagicMock, patch
from src.cost_tracker import CostTracker
# Assuming OpenRouterClient is in src.openrouter_client for mocking
# from src.openrouter_client import OpenRouterClient 

class TestCostTracker(unittest.TestCase):

    def setUp(self):
        """Set up a new CostTracker instance for each test."""
        # Mock OpenRouterClient for tests
        self.mock_openrouter_client = MagicMock()
        # The CostTracker will need to be updated to accept this client
        self.tracker = CostTracker(openrouter_client=self.mock_openrouter_client)

    # --- Existing tests for numeric costs (should still pass) ---
    def test_cost_tracker_creation(self):
        """Test that the cost tracker is created with total_cost = 0."""
        self.assertEqual(self.tracker.get_total_cost(), 0, "Initial cost should be 0.")

    def test_track_cost_single_numeric(self):
        """Test tracking a single numeric cost."""
        self.tracker.track_cost(10)
        self.assertEqual(self.tracker.get_total_cost(), 10)

    def test_track_cost_multiple_numeric(self):
        """Test tracking multiple numeric costs."""
        self.tracker.track_cost(5)
        self.tracker.track_cost(15)
        self.assertEqual(self.tracker.get_total_cost(), 20)

    def test_get_total_cost_initial(self):
        """Test get_total_cost returns 0 initially."""
        self.assertEqual(self.tracker.get_total_cost(), 0)

    def test_get_total_cost_after_tracking(self):
        """Test get_total_cost returns the correct cost after tracking."""
        self.tracker.track_cost(25)
        self.assertEqual(self.tracker.get_total_cost(), 25)

    def test_reset_cost(self):
        """Test reset_cost sets the total cost back to 0."""
        self.tracker.track_cost(50)
        self.tracker.reset_cost()
        self.assertEqual(self.tracker.get_total_cost(), 0)

    def test_reset_cost_on_zero_cost(self):
        """Test reset_cost on a tracker that already has zero cost."""
        self.tracker.reset_cost()
        self.assertEqual(self.tracker.get_total_cost(), 0)
        
    def test_track_cost_non_numeric_simple_dict_no_pricing_info(self):
        """Test tracking simple dict without pricing info adds 0 cost."""
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost({"model": "gpt-4", "tokens": 100}) # Old non-numeric test case
        self.assertEqual(self.tracker.get_total_cost(), initial_cost, "Simple dict without pricing info should add 0 cost.")

    def test_track_cost_other_non_numeric_types(self):
        """Test tracking other non-numeric types (string, list) adds 0 cost."""
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost("some string")
        self.tracker.track_cost([1, 2, 3])
        self.assertEqual(self.tracker.get_total_cost(), initial_cost, "Strings or lists should add 0 cost.")

    # --- Tests for OpenRouterClient integration and token-based costs ---
    def test_track_cost_openrouter_model_success(self):
        """Test successful cost calculation for an openrouter model."""
        model_id = "vendor/model"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id,
            "pricing": {"prompt": "0.0000010", "completion": "0.0000020"} # 1 per M, 2 per M
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 1000000, # 1M tokens
            "completion_tokens": 1000000 # 1M tokens
        }
        self.tracker.track_cost(cost_details)
        # Expected cost: 1M * 1.0/1M + 1M * 2.0/1M = 1.0 + 2.0 = 3.0
        self.assertAlmostEqual(self.tracker.get_total_cost(), 3.0)
        self.mock_openrouter_client.get_model_details.assert_called_once_with(model_id)

    def test_track_cost_openrouter_model_zero_pricing(self):
        """Test openrouter model with zero pricing (free model)."""
        model_id = "vendor/free_model"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id,
            "pricing": {"prompt": "0.0", "completion": "0.0"}
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 1000, "completion_tokens": 200
        }
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), 0.0)

    def test_track_cost_openrouter_model_incomplete_pricing_prompt(self):
        """Test openrouter model with missing prompt price (adds 0 cost)."""
        model_id = "vendor/incomplete_price_model_1"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id,
            "pricing": {"completion": "0.0000020"} # Missing prompt
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 1000, "completion_tokens": 200
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost) # Should add 0

    def test_track_cost_openrouter_model_incomplete_pricing_completion(self):
        """Test openrouter model with missing completion price (adds 0 cost)."""
        model_id = "vendor/incomplete_price_model_2"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id,
            "pricing": {"prompt": "0.0000010"} # Missing completion
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 1000, "completion_tokens": 200
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost) # Should add 0

    def test_track_cost_openrouter_model_missing_pricing_key(self):
        """Test openrouter model with missing 'pricing' key (adds 0 cost)."""
        model_id = "vendor/no_pricing_key_model"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id # 'pricing' key is missing
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 1000, "completion_tokens": 200
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)

    def test_track_cost_openrouter_model_not_found(self):
        """Test openrouter model not found by client (adds 0 cost)."""
        model_id = "vendor/unknown_model"
        self.mock_openrouter_client.get_model_details.return_value = None
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 1000, "completion_tokens": 200
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)
        self.mock_openrouter_client.get_model_details.assert_called_once_with(model_id)
        
    # --- Tests for direct pricing in cost_details ---
    def test_track_cost_direct_pricing_in_details(self):
        """Test cost calculation when pricing is provided directly in cost_details."""
        cost_details = {
            "model_identifier": "any_model/can_be_anything_here",
            "prompt_tokens": 100000,    # 0.1 M tokens
            "completion_tokens": 200000, # 0.2 M tokens
            "cost_per_prompt_token": 0.0000005, # 0.5 per M
            "cost_per_completion_token": 0.0000015 # 1.5 per M
        }
        self.tracker.track_cost(cost_details)
        # Expected cost: 0.1M * 0.5 + 0.2M * 1.5 = 0.05 + 0.30 = 0.35
        self.assertAlmostEqual(self.tracker.get_total_cost(), 0.35)
        self.mock_openrouter_client.get_model_details.assert_not_called() # Should not call client

    def test_track_cost_direct_pricing_incomplete_prompt_price(self):
        """Test direct pricing with missing prompt price (adds 0)."""
        cost_details = {
            "model_identifier": "any_model_direct_incomplete_1",
            "prompt_tokens": 1000, "completion_tokens": 200,
            "cost_per_completion_token": 0.0000015
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)

    def test_track_cost_direct_pricing_incomplete_completion_price(self):
        """Test direct pricing with missing completion price (adds 0)."""
        cost_details = {
            "model_identifier": "any_model_direct_incomplete_2",
            "prompt_tokens": 1000, "completion_tokens": 200,
            "cost_per_prompt_token": 0.0000005
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)

    # --- Tests for non-openrouter models without direct pricing ---
    def test_track_cost_non_openrouter_model_no_direct_pricing(self):
        """Test non-openrouter model without direct pricing (adds 0 cost)."""
        cost_details = {
            "model_identifier": "openai:gpt-4", # Not "openrouter:" and no direct price keys
            "prompt_tokens": 1000, "completion_tokens": 200
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)
        self.mock_openrouter_client.get_model_details.assert_not_called()

    # --- Tests for malformed cost_details ---
    def test_track_cost_malformed_missing_model_identifier(self):
        """Test malformed cost_details: missing model_identifier (adds 0 cost)."""
        cost_details = {
            "prompt_tokens": 1000, "completion_tokens": 200,
            "cost_per_prompt_token": 0.1, "cost_per_completion_token": 0.2 # Has direct pricing
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)

    def test_track_cost_malformed_missing_prompt_tokens_with_direct_price(self):
        """Test malformed: missing prompt_tokens with direct pricing (adds 0)."""
        cost_details = {
            "model_identifier": "some_model_direct_malformed_1",
            "completion_tokens": 200,
            "cost_per_prompt_token": 0.1, "cost_per_completion_token": 0.2
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)

    def test_track_cost_malformed_missing_completion_tokens_with_direct_price(self):
        """Test malformed: missing completion_tokens with direct pricing (adds 0)."""
        cost_details = {
            "model_identifier": "some_model_direct_malformed_2",
            "prompt_tokens": 1000,
            "cost_per_prompt_token": 0.1, "cost_per_completion_token": 0.2
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)

    def test_track_cost_malformed_missing_prompt_tokens_with_openrouter_lookup(self):
        """Test malformed: missing prompt_tokens for openrouter model (adds 0)."""
        model_id = "vendor/model_malformed_1"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id, "pricing": {"prompt": "0.1", "completion": "0.2"}
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "completion_tokens": 200
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)
        # OpenRouterClient might still be called if model_identifier is present and valid prefix
        # self.mock_openrouter_client.get_model_details.assert_called_once_with(model_id)

    def test_track_cost_malformed_missing_completion_tokens_with_openrouter_lookup(self):
        """Test malformed: missing completion_tokens for openrouter model (adds 0)."""
        model_id = "vendor/model_malformed_2"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id, "pricing": {"prompt": "0.1", "completion": "0.2"}
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 1000
        }
        initial_cost = self.tracker.get_total_cost()
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), initial_cost)
        # self.mock_openrouter_client.get_model_details.assert_called_once_with(model_id)

    def test_track_cost_zero_tokens_openrouter(self):
        """Test zero tokens for an openrouter model results in zero cost."""
        model_id = "vendor/zero_token_model"
        self.mock_openrouter_client.get_model_details.return_value = {
            "id": model_id,
            "pricing": {"prompt": "0.0000010", "completion": "0.0000020"}
        }
        cost_details = {
            "model_identifier": f"openrouter:{model_id}",
            "prompt_tokens": 0,
            "completion_tokens": 0
        }
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), 0.0)
        self.mock_openrouter_client.get_model_details.assert_called_once_with(model_id)

    def test_track_cost_zero_tokens_direct_pricing(self):
        """Test zero tokens with direct pricing results in zero cost."""
        cost_details = {
            "model_identifier": "any_model_zero_tokens",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_per_prompt_token": 0.0000005,
            "cost_per_completion_token": 0.0000015
        }
        self.tracker.track_cost(cost_details)
        self.assertAlmostEqual(self.tracker.get_total_cost(), 0.0)


if __name__ == '__main__':
    unittest.main()
