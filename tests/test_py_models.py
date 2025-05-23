import unittest
from pydantic import ValidationError
from py_models.pd_reader_model import PDReaderModel
from py_models.weather_model import WeatherModel
from py_models.general_example_model import GeneralExampleModel

class TestPyModels(unittest.TestCase):
    def test_pd_reader_model_creation(self):
        """Test PDReaderModel creation with valid data"""
        model = PDReaderModel(
            content="PDF content here",
            num_pages=10,
            metadata={"author": "John Doe", "title": "Test PDF"},
            extracted_text="Extracted text from PDF"
        )
        
        self.assertEqual(model.content, "PDF content here")
        self.assertEqual(model.num_pages, 10)
        self.assertEqual(model.metadata["author"], "John Doe")
        self.assertEqual(model.extracted_text, "Extracted text from PDF")
    
    def test_pd_reader_model_optional_fields(self):
        """Test PDReaderModel with optional fields"""
        model = PDReaderModel()
        
        self.assertIsNone(model.content)
        self.assertIsNone(model.num_pages)
        self.assertIsNone(model.metadata)
        self.assertIsNone(model.extracted_text)
    
    def test_weather_model_creation(self):
        """Test WeatherModel creation with valid data"""
        model = WeatherModel(
            location="Sofia, Bulgaria",
            temperature=20.5,
            conditions="Sunny"
        )
        
        self.assertEqual(model.location, "Sofia, Bulgaria")
        self.assertEqual(model.temperature, 20.5)
        self.assertEqual(model.conditions, "Sunny")
    
    def test_weather_model_optional_fields(self):
        """Test WeatherModel with optional fields"""
        model = WeatherModel()
        
        self.assertIsNone(model.location)
        self.assertIsNone(model.temperature)
        self.assertIsNone(model.conditions)
    
    def test_general_example_model_creation(self):
        """Test GeneralExampleModel creation with valid data"""
        model = GeneralExampleModel(
            content="Some content",
            extracted_data={"key1": "value1"},
            key="dog",
            value="Roger"
        )
        
        self.assertEqual(model.content, "Some content")
        self.assertEqual(model.extracted_data["key1"], "value1")
        self.assertEqual(model.key, "dog")
        self.assertEqual(model.value, "Roger")
    
    def test_general_example_model_optional_fields(self):
        """Test GeneralExampleModel with optional fields"""
        model = GeneralExampleModel()
        
        self.assertIsNone(model.content)
        self.assertIsNone(model.extracted_data)
        self.assertIsNone(model.key)
        self.assertIsNone(model.value)
    
    def test_pd_reader_model_validation(self):
        """Test PDReaderModel validation"""
        # Test with invalid num_pages type
        with self.assertRaises(ValidationError):
            PDReaderModel(num_pages="not a number")
        
        # Test with valid partial data
        model = PDReaderModel(content="Only content")
        self.assertEqual(model.content, "Only content")
        self.assertIsNone(model.num_pages)
    
    def test_weather_model_validation(self):
        """Test WeatherModel validation"""
        # Test with invalid temperature type
        with self.assertRaises(ValidationError):
            WeatherModel(temperature="not a float")
        
        # Test with valid partial data
        model = WeatherModel(location="Sofia")
        self.assertEqual(model.location, "Sofia")
        self.assertIsNone(model.temperature)
    
    def test_model_json_serialization(self):
        """Test model JSON serialization"""
        weather = WeatherModel(
            location="Sofia",
            temperature=20.5,
            conditions="Sunny"
        )
        
        # Test JSON dump
        json_str = weather.model_dump_json()
        self.assertIn('"location":"Sofia"', json_str)
        self.assertIn('"temperature":20.5', json_str)
        
        # Test JSON parse
        weather2 = WeatherModel.model_validate_json(json_str)
        self.assertEqual(weather2.location, weather.location)
        self.assertEqual(weather2.temperature, weather.temperature)
        self.assertEqual(weather2.conditions, weather.conditions)
    
    def test_model_dict_conversion(self):
        """Test model dict conversion"""
        general = GeneralExampleModel(
            key="test",
            value="value"
        )
        
        # Convert to dict
        data = general.model_dump()
        self.assertEqual(data["key"], "test")
        self.assertEqual(data["value"], "value")
        self.assertIsNone(data["content"])
        self.assertIsNone(data["extracted_data"])
        
        # Create from dict
        general2 = GeneralExampleModel.model_validate(data)
        self.assertEqual(general2.key, general.key)
        self.assertEqual(general2.value, general.value)

if __name__ == '__main__':
    unittest.main()
