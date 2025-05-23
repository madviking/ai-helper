import unittest
from unittest.mock import patch # Import patch
from src.cost_tracker import CostTracker, MODEL_PRICING

class TestCostTracker(unittest.TestCase):
    def setUp(self):
        # Use a fresh CostTracker for each test, potentially with specific pricing for some tests
        self.tracker = CostTracker()

    def test_cost_tracker_creation(self):
        self.assertEqual(self.tracker.get_total_cost(), 0.0)
        self.assertIsNotNone(self.tracker.model_pricing)

    def test_calculate_cost_known_model(self):
        # gpt-3.5-turbo: input $0.50/1M, output $1.50/1M
        # Per token: input 0.0000005, output 0.0000015
        prompt_tokens = 1000
        completion_tokens = 500
        expected_cost = (prompt_tokens * 0.0000005) + (completion_tokens * 0.0000015)
        # 1000 * 0.0000005 = 0.0005
        # 500 * 0.0000015 = 0.00075
        # Total = 0.00125
        
        cost = self.tracker.calculate_cost("gpt-3.5-turbo", prompt_tokens, completion_tokens)
        self.assertAlmostEqual(cost, expected_cost, places=7)

        cost_with_provider = self.tracker.calculate_cost("openai:gpt-3.5-turbo", prompt_tokens, completion_tokens)
        self.assertAlmostEqual(cost_with_provider, expected_cost, places=7)
        
        cost_with_org_model = self.tracker.calculate_cost("openai/gpt-3.5-turbo", prompt_tokens, completion_tokens)
        self.assertAlmostEqual(cost_with_org_model, expected_cost, places=7)


    def test_calculate_cost_unknown_model_uses_default(self):
        # Default: input $20/1M, output $20/1M
        # Per token: input 0.000020, output 0.000020
        prompt_tokens = 100
        completion_tokens = 50
        expected_cost = (prompt_tokens * 0.000020) + (completion_tokens * 0.000020)
        # 100 * 0.000020 = 0.002
        # 50 * 0.000020 = 0.001
        # Total = 0.003
        
        with patch('builtins.print') as mock_print: # Suppress warning print
            cost = self.tracker.calculate_cost("unknown-model-123", prompt_tokens, completion_tokens)
            self.assertAlmostEqual(cost, expected_cost, places=7)
            mock_print.assert_any_call("Warning: Pricing not found for model 'unknown-model-123' (normalized: 'unknown-model-123'). Using default if available, or cannot calculate cost.")

    def test_calculate_cost_no_default_no_match(self):
        # Test with pricing data that has no "default"
        custom_pricing = {"specific-model": {"input": 0.1, "output": 0.1}}
        tracker_no_default = CostTracker(pricing_data=custom_pricing)
        with patch('builtins.print') as mock_print:
            cost = tracker_no_default.calculate_cost("another-unknown-model", 100, 100)
            self.assertIsNone(cost)
            mock_print.assert_any_call("Warning: Pricing not found for model 'another-unknown-model' (normalized: 'another-unknown-model'). Using default if available, or cannot calculate cost.")


    def test_track_cost_with_pre_calculated_cost(self):
        self.tracker.track_cost({"cost": 0.005})
        self.assertEqual(self.tracker.get_total_cost(), 0.005)
        self.tracker.track_cost({"cost": 0.010})
        self.assertEqual(self.tracker.get_total_cost(), 0.015)

    def test_track_cost_with_token_counts(self):
        # gpt-4o: input $5/1M (0.000005), output $15/1M (0.000015)
        details1 = {
            "model_name": "gpt-4o",
            "prompt_tokens": 10000, # 10k * 0.000005 = 0.05
            "completion_tokens": 20000 # 20k * 0.000015 = 0.30
        } # Total = 0.35
        expected_cost1 = 0.35
        self.tracker.track_cost(details1)
        self.assertAlmostEqual(self.tracker.get_total_cost(), expected_cost1, places=7)

        # gpt-3.5-turbo
        details2 = {
            "model_name": "openai:gpt-3.5-turbo",
            "prompt_tokens": 5000, # 5k * 0.0000005 = 0.0025
            "completion_tokens": 1000 # 1k * 0.0000015 = 0.0015
        } # Total = 0.0040
        expected_cost2 = 0.0040
        self.tracker.track_cost(details2)
        self.assertAlmostEqual(self.tracker.get_total_cost(), expected_cost1 + expected_cost2, places=7)

    def test_track_cost_insufficient_details(self):
        with patch('builtins.print') as mock_print:
            self.tracker.track_cost({"model_name": "gpt-4o"}) # Missing tokens
            mock_print.assert_called_with("Warning: Insufficient details to track cost: {'model_name': 'gpt-4o'}")
        self.assertEqual(self.tracker.get_total_cost(), 0.0)

    def test_track_cost_unknown_model_no_default_pricing(self):
        custom_pricing = {"specific-model": {"input": 0.1, "output": 0.1}}
        tracker_no_default = CostTracker(pricing_data=custom_pricing)
        details = {
            "model_name": "very-unknown-model",
            "prompt_tokens": 100,
            "completion_tokens": 100
        }
        with patch('builtins.print') as mock_print:
            tracker_no_default.track_cost(details)
            # First print from calculate_cost, second from track_cost itself
            mock_print.assert_any_call("Warning: Pricing not found for model 'very-unknown-model' (normalized: 'very-unknown-model'). Using default if available, or cannot calculate cost.")
            # Since calculate_cost returns None, track_cost prints "Could not calculate..."
            mock_print.assert_any_call("Could not calculate cost for: {'model_name': 'very-unknown-model', 'prompt_tokens': 100, 'completion_tokens': 100}")

        self.assertEqual(tracker_no_default.get_total_cost(), 0.0)


    def test_get_total_cost(self):
        self.assertEqual(self.tracker.get_total_cost(), 0.0)
        self.tracker.track_cost({"cost": 0.001})
        self.assertEqual(self.tracker.get_total_cost(), 0.001)

    def test_reset_cost(self):
        self.tracker.track_cost({"cost": 0.0025})
        self.assertEqual(self.tracker.get_total_cost(), 0.0025)
        self.tracker.reset_cost()
        self.assertEqual(self.tracker.get_total_cost(), 0.0)

if __name__ == '__main__':
    unittest.main()
