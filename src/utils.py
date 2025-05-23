import os
from dotenv import load_dotenv

class MissingEnvVariableError(Exception):
    """Custom exception for missing environment variables."""
    pass

def get_env_variable(variable_name: str) -> str:
    """
    Loads environment variables from .env file and retrieves a specific variable.

    Args:
        variable_name (str): The name of the environment variable to retrieve.

    Returns:
        str: The value of the environment variable.

    Raises:
        MissingEnvVariableError: If the environment variable is not found.
    """
    load_dotenv()  # Load variables from .env file
    
    value = os.environ.get(variable_name)
    
    if value is None:
        raise MissingEnvVariableError(
            f"{variable_name} not found in environment variables. "
            "Please ensure it's set in your .env file or environment."
        )
    return value
