def calculator(expression: str):
    """A simple calculator that can add, subtract, multiply, and divide."""
    try:
        # Evaluate the expression. Using eval() can be risky with untrusted input,
        # but for this example, we'll assume the input is controlled.
        result = eval(expression)
        return result
    except Exception as e:
        raise ValueError(f"Invalid expression: {expression}") from e

def weather(location: str):
    """A tool to get the current weather information."""
    # TODO: Implement actual weather API call
    print(f"Getting weather for: {location}")
    return f"Weather information for {location} (Not implemented)"

def pdf_reader(file_path: str):
    """A tool to read and extract information from a PDF file."""
    # TODO: Implement actual PDF reading logic
    print(f"Reading PDF file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        # Basic implementation reads as text, will need a proper PDF library
        print(f"Warning: Could not read PDF as text: {e}")
        return f"Could not read PDF content from {file_path} (Not fully implemented)"
