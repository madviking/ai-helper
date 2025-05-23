import unittest
from unittest.mock import patch, MagicMock
import os

# Assuming the utils module is in src.utils
from src.utils import get_env_variable, MissingEnvVariableError

class TestGetEnvVariable(unittest.TestCase):

    @patch('src.utils.load_dotenv')  # Patch load_dotenv in the utils module
    @patch.dict(os.environ, {"TEST_VAR_EXISTS": "test_value"})
    def test_get_env_variable_exists(self, mock_load_dotenv):
        """Test retrieving an existing environment variable."""
        variable_name = "TEST_VAR_EXISTS"
        expected_value = "test_value"
        
        value = get_env_variable(variable_name)
        
        self.assertEqual(value, expected_value)
        mock_load_dotenv.assert_called_once()

    @patch('src.utils.load_dotenv')
    @patch.dict(os.environ, {}, clear=True) # Start with a clean os.environ for this test
    def test_get_env_variable_not_exists_raises_error(self, mock_load_dotenv):
        """Test that MissingEnvVariableError is raised for a non-existent variable."""
        variable_name = "TEST_VAR_DOES_NOT_EXIST"
        
        with self.assertRaises(MissingEnvVariableError) as context:
            get_env_variable(variable_name)
        
        self.assertIn(f"{variable_name} not found", str(context.exception))
        mock_load_dotenv.assert_called_once()

    @patch('src.utils.load_dotenv')
    @patch.dict(os.environ, {"ANOTHER_TEST_VAR": "another_value"})
    def test_load_dotenv_called(self, mock_load_dotenv):
        """Test that load_dotenv is called when get_env_variable is invoked."""
        get_env_variable("ANOTHER_TEST_VAR")
        mock_load_dotenv.assert_called_once()

    @patch('src.utils.load_dotenv')
    @patch.dict(os.environ, {"EMPTY_VAR": ""}) # Variable exists but is an empty string
    def test_get_env_variable_empty_value(self, mock_load_dotenv):
        """Test retrieving an existing environment variable that has an empty string value."""
        variable_name = "EMPTY_VAR"
        expected_value = ""
        
        value = get_env_variable(variable_name)
        
        self.assertEqual(value, expected_value)
        mock_load_dotenv.assert_called_once()

if __name__ == '__main__':
    unittest.main()
