import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import os
from datetime import datetime, timedelta
from src.cost_tracker import CostTracker

class TestCostTracker(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.cost_tracker = CostTracker()
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove any test files created
        test_files = ['py_model_cost.json', 'llm_model_cost.json', 'total_cost.json', 'models.json']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    def test_cost_tracker_initialization(self):
        """Test CostTracker initialization"""
        self.assertEqual(self.cost_tracker._total_cost, 0)
        self.assertIsInstance(self.cost_tracker._cost_info, dict)
    
    @patch('requests.get')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    def test_init_cost_info_fetch_from_api(self, mock_getmtime, mock_exists, mock_file, mock_get):
        """Test fetching cost info from OpenRouter API"""
        # Mock file doesn't exist or is old
        mock_exists.return_value = False
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'id': 'openai/gpt-3.5-turbo',
                    'pricing': {
                        'prompt': '0.0005',
                        'completion': '0.0015'
                    }
                },
                {
                    'id': 'anthropic/claude-3',
                    'pricing': {
                        'prompt': '0.003',
                        'completion': '0.015'
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Create new instance to trigger init
        tracker = CostTracker()
        
        # Verify API was called
        mock_get.assert_called_once_with('https://openrouter.ai/api/v1/models')
        
        # Verify file was written
        mock_file.assert_called()
        
        # Verify data was processed correctly
        self.assertIn('openai/gpt-3.5-turbo', tracker._cost_info)
        self.assertEqual(tracker._cost_info['openai/gpt-3.5-turbo']['pricing']['prompt'], '0.0005')
        self.assertEqual(tracker._cost_info['openai/gpt-3.5-turbo']['pricing']['completion'], '0.0015')
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    def test_init_cost_info_use_cached(self, mock_getmtime, mock_exists, mock_file):
        """Test using cached cost info when recent"""
        # Mock file exists and is recent
        mock_exists.return_value = True
        mock_getmtime.return_value = datetime.now().timestamp()
        
        # Mock cached data
        cached_data = {
            'data': [
                {
                    'id': 'openai/gpt-4',
                    'pricing': {
                        'prompt': '0.03',
                        'completion': '0.06'
                    }
                }
            ]
        }
        mock_file.return_value.read.return_value = json.dumps(cached_data)
        
        # Create new instance
        tracker = CostTracker()
        
        # Verify cached data was loaded
        self.assertIn('openai/gpt-4', tracker._cost_info)
    
    def test_add_cost_calculation(self):
        """Test cost calculation and tracking"""
        # Set up mock cost info
        self.cost_tracker._cost_info = {
            'openai/gpt-3.5-turbo': {
                'pricing': {
                    'prompt': '0.0005',
                    'completion': '0.0015'
                }
            }
        }
        
        # Add cost
        self.cost_tracker.add_cost(
            input_tokens=1000,
            output_tokens=500,
            pydantic_model='WeatherModel',
            llm_model='openai/gpt-3.5-turbo'
        )
        
        # Calculate expected cost
        # Input: 1000 tokens * $0.0005/1000 = $0.0005
        # Output: 500 tokens * $0.0015/1000 = $0.00075
        # Total: $0.00125
        expected_cost = 0.00125
        
        self.assertAlmostEqual(self.cost_tracker.get_total_cost(), expected_cost, places=6)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_add_cost_updates_files(self, mock_file):
        """Test that add_cost updates all tracking files"""
        # Set up mock cost info
        self.cost_tracker._cost_info = {
            'openai/gpt-3.5-turbo': {
                'pricing': {
                    'prompt': '0.0005',
                    'completion': '0.0015'
                }
            }
        }
        
        # Add cost
        self.cost_tracker.add_cost(
            input_tokens=1000,
            output_tokens=500,
            pydantic_model='WeatherModel',
            llm_model='openai/gpt-3.5-turbo'
        )
        
        # Verify files were written
        file_calls = [call[0][0] for call in mock_file.call_args_list]
        self.assertIn('py_model_cost.json', file_calls)
        self.assertIn('llm_model_cost.json', file_calls)
        self.assertIn('total_cost.json', file_calls)
    
    def test_add_cost_unknown_model(self):
        """Test adding cost for unknown model"""
        # Try to add cost for unknown model
        self.cost_tracker.add_cost(
            input_tokens=1000,
            output_tokens=500,
            pydantic_model='TestModel',
            llm_model='unknown/model'
        )
        
        # Should not raise error, cost should remain 0
        self.assertEqual(self.cost_tracker.get_total_cost(), 0)
    
    def test_track_cost_method(self):
        """Test the track_cost method"""
        cost_details = {
            'model': 'openai/gpt-3.5-turbo',
            'input_tokens': 100,
            'output_tokens': 50,
            'cost': 0.0001
        }
        
        # Track cost
        self.cost_tracker.track_cost(cost_details)
        
        # Verify cost was added
        self.assertGreater(self.cost_tracker.get_total_cost(), 0)
    
    def test_reset_cost(self):
        """Test resetting cost"""
        # Add some cost
        self.cost_tracker._total_cost = 10.50
        
        # Reset
        self.cost_tracker.reset_cost()
        
        # Verify reset
        self.assertEqual(self.cost_tracker.get_total_cost(), 0)
    
    def test_get_total_cost(self):
        """Test getting total cost"""
        # Set a specific cost
        self.cost_tracker._total_cost = 5.25
        
        # Get cost
        total = self.cost_tracker.get_total_cost()
        
        self.assertEqual(total, 5.25)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_cost_persistence(self, mock_exists, mock_file):
        """Test that costs are persisted across instances"""
        # Mock existing cost files
        mock_exists.return_value = True
        
        # Mock file contents
        cost_data = {
            'total': 15.75,
            'models': {
                'WeatherModel': 5.25,
                'GeneralExampleModel': 10.50
            }
        }
        mock_file.return_value.read.return_value = json.dumps(cost_data)
        
        # Create new tracker
        tracker = CostTracker()
        
        # Should load existing costs
        # Note: Implementation needs to support loading existing costs

if __name__ == '__main__':
    unittest.main()
