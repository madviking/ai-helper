import unittest
from unittest.mock import patch, MagicMock
import os
from src.tools import calculator, weather, pdf_reader

class TestTools(unittest.TestCase):
    def test_calculator_tool_addition(self):
        """Test calculator with addition"""
        result = calculator("2 + 3")
        self.assertEqual(result, 5)
    
    def test_calculator_tool_subtraction(self):
        """Test calculator with subtraction"""
        result = calculator("10 - 4")
        self.assertEqual(result, 6)
    
    def test_calculator_tool_multiplication(self):
        """Test calculator with multiplication"""
        result = calculator("5 * 6")
        self.assertEqual(result, 30)
    
    def test_calculator_tool_division(self):
        """Test calculator with division"""
        result = calculator("20 / 4")
        self.assertEqual(result, 5.0)
    
    def test_calculator_tool_complex_expression(self):
        """Test calculator with complex expression"""
        result = calculator("(10 + 5) * 2 - 8 / 4")
        self.assertEqual(result, 28.0)
    
    def test_calculator_tool_invalid_expression(self):
        """Test calculator with invalid expression"""
        with self.assertRaises(Exception):
            calculator("invalid expression")
    
    @patch('requests.get')
    @patch.dict(os.environ, {'WEATHER_API_KEY': 'test_key'})
    def test_weather_tool_success(self, mock_get):
        """Test weather tool with successful API response"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'location': {
                'name': 'Sofia',
                'country': 'Bulgaria'
            },
            'current': {
                'temp_c': 20,
                'condition': {
                    'text': 'Sunny'
                }
            }
        }
        mock_get.return_value = mock_response
        
        result = weather("Sofia")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['location'], 'Sofia, Bulgaria')
        self.assertEqual(result['temperature'], 20)
        self.assertEqual(result['conditions'], 'Sunny')
        
        # Verify API was called correctly
        mock_get.assert_called_once_with(
            'http://api.weatherapi.com/v1/current.json',
            params={'key': 'test_key', 'q': 'Sofia', 'aqi': 'no'}
        )
    
    @patch('requests.get')
    @patch.dict(os.environ, {'WEATHER_API_KEY': 'test_key'})
    def test_weather_tool_api_error(self, mock_get):
        """Test weather tool with API error"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {'error': {'message': 'Location not found'}}
        mock_get.return_value = mock_response
        
        with self.assertRaises(Exception) as context:
            weather("InvalidLocation")
        
        self.assertIn("Weather API error", str(context.exception))
    
    @patch.dict(os.environ, {}, clear=True)
    def test_weather_tool_no_api_key(self):
        """Test weather tool without API key"""
        with self.assertRaises(Exception) as context:
            weather("Sofia")
        
        self.assertIn("WEATHER_API_KEY", str(context.exception))
    
    def test_pdf_reader_tool_success(self):
        """Test PDF reader with valid PDF file"""
        # Create a test PDF file path
        test_pdf_path = "tests/files/test.pdf"
        
        # Assuming the PDF exists and contains text
        if os.path.exists(test_pdf_path):
            result = pdf_reader(test_pdf_path)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
        else:
            # Skip test if test file doesn't exist
            self.skipTest(f"Test PDF file {test_pdf_path} not found")
    
    def test_pdf_reader_tool_file_not_found(self):
        """Test PDF reader with non-existent file"""
        with self.assertRaises(FileNotFoundError):
            pdf_reader("non_existent_file.pdf")
    
    def test_pdf_reader_tool_invalid_file_type(self):
        """Test PDF reader with non-PDF file"""
        # Create a temporary text file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a PDF")
            temp_file = f.name
        
        try:
            with self.assertRaises(Exception) as context:
                pdf_reader(temp_file)
            self.assertIn("PDF", str(context.exception))
        finally:
            os.unlink(temp_file)

if __name__ == '__main__':
    unittest.main()
