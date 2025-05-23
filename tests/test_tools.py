import unittest
import json
import requests # Import requests for requests.exceptions
from unittest.mock import patch, MagicMock
from src.tools import calculator, weather, pdf_reader

class TestTools(unittest.TestCase):
    def test_calculator_tool_basic_operations(self):
        self.assertEqual(calculator("2 + 2"), "4")
        self.assertEqual(calculator("5 - 3"), "2")
        self.assertEqual(calculator("4 * 3"), "12")
        self.assertEqual(calculator("10 / 2"), "5.0") # eval results in float for division
        self.assertEqual(calculator("2 + 3 * 4"), "14") # Order of operations
        self.assertEqual(calculator("(2 + 3) * 4"), "20")
        self.assertEqual(calculator("10 / 0"), "Error evaluating expression: division by zero")

    def test_calculator_tool_invalid_expression(self):
        # After regex update in src/tools.py: ^(?!.*[\+\-\*\/]{2})[0-9\.\+\-\*\/\(\)\s]+$
        self.assertEqual(calculator("2 ++ 2"), "Error: Invalid characters or format in expression.")
        self.assertEqual(calculator("abc"), "Error: Invalid characters or format in expression.")
        self.assertEqual(calculator("import os"), "Error: Invalid characters or format in expression.") # Basic security check

    @patch('src.tools.requests.get')
    @patch('src.tools.WEATHER_API_KEY', "fake_api_key") # Patch the global var in the module
    def test_weather_tool_success(self, mock_requests_get):
        # mock_getenv is no longer needed here as we patch WEATHER_API_KEY directly
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "location": {"name": "London"},
            "current": {
                "temp_c": 15,
                "condition": {"text": "Partly cloudy"},
                "humidity": 70
            }
        }
        mock_requests_get.return_value = mock_response

        result_str = weather("London")
        self.assertIsInstance(result_str, str)
        result_json = json.loads(result_str)

        expected_json = {
            "location": "London",
            "condition": "Partly cloudy",
            "temperature_celsius": 15,
            "humidity_percent": 70
        }
        self.assertEqual(result_json, expected_json)
        mock_requests_get.assert_called_once_with(
            "http://api.weatherapi.com/v1/current.json",
            params={"key": "fake_api_key", "q": "London", "aqi": "no"}
        )

    @patch('src.tools.WEATHER_API_KEY', None) # Patch the global var in the module
    def test_weather_tool_no_api_key(self):
        # mock_getenv is no longer needed
        self.assertEqual(weather("London"), "Error: WEATHER_API_KEY not configured.")

    @patch('src.tools.WEATHER_API_KEY', "fake_api_key") # Ensure API key is "present"
    def test_weather_tool_no_location(self):
        self.assertEqual(weather(""), "Error: Location not specified.")

    @patch('src.tools.requests.get')
    @patch('src.tools.WEATHER_API_KEY', "fake_api_key") # Patch the global var
    def test_weather_tool_api_error(self, mock_requests_get):
        # mock_getenv is no longer needed
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error")
        mock_requests_get.return_value = mock_response

        result = weather("London")
        self.assertTrue("Error fetching weather data" in result)
        self.assertTrue("API Error" in result)
        
    @patch('src.tools.requests.get')
    @patch('src.tools.WEATHER_API_KEY', "fake_api_key") # Patch the global var
    def test_weather_tool_json_decode_error(self, mock_requests_get):
        # mock_getenv is no longer needed
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_requests_get.return_value = mock_response
        
        self.assertEqual(weather("London"), "Error: Could not parse weather API response.")

    def test_pdf_reader_tool_placeholder(self):
        # Test the placeholder implementation
        result = pdf_reader("dummy/path/to/file.pdf")
        self.assertEqual(result, "Placeholder: PDF reader for dummy/path/to/file.pdf. Text extraction not yet implemented.")

if __name__ == '__main__':
    unittest.main()
