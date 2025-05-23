import unittest
from src.cost_tracker import CostTracker

class TestCostTracker(unittest.TestCase):
    def test_cost_tracker_creation(self):
        tracker = CostTracker()
        self.assertEqual(tracker.get_total_cost(), 0)

    def test_track_cost(self):
        tracker = CostTracker()
        tracker.track_cost(10, 0, "dummy_model")
        self.assertEqual(tracker.get_total_cost(), 10)
        tracker.track_cost(5, 0, "dummy_model")
        self.assertEqual(tracker.get_total_cost(), 15)

    def test_get_total_cost(self):
        tracker = CostTracker()
        self.assertEqual(tracker.get_total_cost(), 0)
        tracker.track_cost(25, 0, "dummy_model")
        self.assertEqual(tracker.get_total_cost(), 25)

    def test_reset_cost(self):
        tracker = CostTracker()
        tracker.track_cost(100, 0, "dummy_model")
        self.assertEqual(tracker.get_total_cost(), 100)
        tracker.reset_cost()
        self.assertEqual(tracker.get_total_cost(), 0)

if __name__ == '__main__':
    unittest.main()
