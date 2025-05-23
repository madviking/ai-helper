import re

def calculator(expression: str):
    """
    A simple calculator that can add, subtract, multiply, and divide.
    Parses expressions like "number operator number".
    Supports integers and floating-point numbers, including negative numbers.
    """
    # Regex to capture:
    # Group 1: First number (optional sign, digits, optional decimal part)
    # Group 2: Operator (+, -, *, /)
    # Group 3: Second number (optional sign, digits, optional decimal part)
    # Allows optional spaces around numbers and operator.
    pattern = r"^\s*([+-]?\d+\.?\d*)\s*([+\-*/])\s*([+-]?\d+\.?\d*)\s*$"
    
    match = re.fullmatch(pattern, expression)
    
    if not match:
        raise ValueError("Invalid expression format. Expected 'number operator number'.")
        
    num1_str, operator, num2_str = match.groups()
    
    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        raise ValueError("Invalid number format.")
        
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError("division by zero")
        return num1 / num2
    else:
        # This case should ideally be caught by the regex, but as a fallback:
        raise ValueError(f"Unsupported operator: {operator}")

# Placeholder for other tools
def weather(location: str):
    """A tool to get the current weather information."""
    # To be implemented
    pass

def pdf_reader(file_path: str):
    """A tool to read and extract information from a PDF file."""
    # To be implemented
    pass
