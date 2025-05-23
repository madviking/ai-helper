import unittest
from unittest.mock import patch, MagicMock
import time
import requests # Import actual requests for its exceptions

# Assuming OpenRouterClient will be in src.openrouter_client
from src.openrouter_client import OpenRouterClient, OpenRouterAPIError

class TestOpenRouterClient(unittest.TestCase):

    def setUp(self):
        self.client = OpenRouterClient()
        self.mock_models_data = [{"id": "model1", "name": "Model One"}, {"id": "model2", "name": "Model Two"}]
        self.mock_api_response = {"data": self.mock_models_data}

    @patch('src.openrouter_client.requests.get')
    @patch('src.openrouter_client.time.time', return_value=1000) # Mock current time
    def test_get_models_fetch_success(self, mock_time, mock_requests_get):
        """Test successful fetching and parsing of models data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_api_response
        mock_requests_get.return_value = mock_response

        models = self.client.get_models()

        mock_requests_get.assert_called_once_with(OpenRouterClient.MODELS_URL, timeout=10) # Added timeout
        self.assertEqual(models, self.mock_models_data)
        self.assertEqual(self.client._cached_models_data, self.mock_models_data)
        self.assertEqual(self.client._last_fetch_time, 1000) # Check that fetch time is updated

    @patch('src.openrouter_client.requests.get')
    def test_get_models_fetch_http_error(self, mock_requests_get):
        """Test handling of HTTP error during fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not Found")
        mock_requests_get.return_value = mock_response
        
        # Need to import requests for the exception for this test.
        # This assumes `_fetch_from_api` will import requests.
        # If the exception is caught and re-raised as OpenRouterAPIError, this will change.
        # For now, let's assume _fetch_from_api is:
        # try: response.raise_for_status() except requests.exceptions.HTTPError as e: raise OpenRouterAPIError(e)

        # Re-patching requests locally if needed for the exception
        # Actually, the client should raise OpenRouterAPIError
        with self.assertRaises(OpenRouterAPIError):
            self.client.get_models()

    @patch('src.openrouter_client.time.time')
    @patch('src.openrouter_client.requests.get')
    def test_get_models_caching_logic_within_duration(self, mock_requests_get, mock_time):
        """Test that cached data is returned if within cache duration."""
        # Initial fetch
        mock_time.return_value = 1000  # Initial fetch time
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_api_response
        mock_requests_get.return_value = mock_response
        
        self.client.get_models() # Populates cache
        mock_requests_get.assert_called_once() # Called once for initial fetch
        
        # Second call within cache duration
        mock_time.return_value = 1000 + OpenRouterClient.CACHE_DURATION_SECONDS - 1 # Still within cache
        cached_models = self.client.get_models()
        
        self.assertEqual(cached_models, self.mock_models_data)
        mock_requests_get.assert_called_once() # Still only called once (no new HTTP request)

    @patch('src.openrouter_client.time.time')
    @patch('src.openrouter_client.requests.get')
    def test_get_models_caching_logic_expired(self, mock_requests_get, mock_time):
        """Test that a new HTTP request is made if cache has expired."""
        # Initial fetch
        mock_time.return_value = 1000
        first_response_mock = MagicMock()
        first_response_mock.status_code = 200
        first_response_mock.json.return_value = {"data": [{"id": "old_model"}]}
        mock_requests_get.return_value = first_response_mock

        self.client.get_models()
        mock_requests_get.assert_called_once()
        self.assertEqual(self.client._last_fetch_time, 1000)

        # Second call after cache duration
        mock_time.return_value = 1000 + OpenRouterClient.CACHE_DURATION_SECONDS + 1 # Cache expired
        second_response_mock = MagicMock()
        second_response_mock.status_code = 200
        second_response_mock.json.return_value = self.mock_api_response # New data
        mock_requests_get.return_value = second_response_mock

        new_models = self.client.get_models()

        self.assertEqual(new_models, self.mock_models_data)
        self.assertEqual(mock_requests_get.call_count, 2) # Called again for new fetch
        self.assertEqual(self.client._last_fetch_time, 1000 + OpenRouterClient.CACHE_DURATION_SECONDS + 1)


    @patch('src.openrouter_client.requests.get')
    @patch('src.openrouter_client.time.time', return_value=2000)
    def test_get_model_details_success(self, mock_time, mock_requests_get):
        """Test retrieving details for a specific model ID successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_api_response
        mock_requests_get.return_value = mock_response

        # get_models() will be called internally by get_model_details if cache is empty/stale
        model_detail = self.client.get_model_details("model1")
        self.assertEqual(model_detail, self.mock_models_data[0])
        mock_requests_get.assert_called_once() # Ensure data was fetched

        # Test getting another model from already cached data
        model_detail_2 = self.client.get_model_details("model2")
        self.assertEqual(model_detail_2, self.mock_models_data[1])
        mock_requests_get.assert_called_once() # Still only one fetch

    @patch('src.openrouter_client.requests.get')
    @patch('src.openrouter_client.time.time', return_value=3000)
    def test_get_model_details_not_found(self, mock_time, mock_requests_get):
        """Test retrieving details for a non-existent model ID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_api_response
        mock_requests_get.return_value = mock_response

        # Assuming get_model_details should return None if not found
        # The implementation might choose to raise KeyError/ValueError instead.
        # For this test, we'll assume it returns None.
        model_detail = self.client.get_model_details("non_existent_model")
        self.assertIsNone(model_detail)
        mock_requests_get.assert_called_once() # Data fetched

    @patch('src.openrouter_client.requests.get')
    @patch('src.openrouter_client.time.time', return_value=4000)
    def test_get_model_details_empty_cache_and_api_error(self, mock_time, mock_requests_get):
        """Test get_model_details when cache is empty and API call fails."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        mock_requests_get.return_value = mock_response

        # If get_models (called by get_model_details) fails, it should propagate the error
        with self.assertRaises(OpenRouterAPIError):
            self.client.get_model_details("any_model_id")
        mock_requests_get.assert_called_once()

    # Removed the local mock 'requests' class, will use the actual requests.exceptions.HTTPError

if __name__ == '__main__':
    unittest.main()
