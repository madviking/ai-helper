import unittest
from src.tools import calculator # Assuming calculator will be in src.tools
# Placeholders for other tools if they exist in src.tools
# from src.tools import weather, pdf_reader 

# Keep existing TestTools class if it's for other tools, or remove if focusing only on calculator for now
# For this subtask, let's focus on calculator and create a specific test class for it.
# If TestTools was meant for calculator, I'll integrate there. Given the name, it's general.
# I will add a new class TestCalculator.

class TestCalculator(unittest.TestCase):
    def test_add_simple(self):
        self.assertEqual(calculator("2 + 3"), 5)

    def test_add_float(self):
        self.assertEqual(calculator("2.5 + 1.5"), 4.0)

    def test_subtract_simple(self):
        self.assertEqual(calculator("5 - 2"), 3)

    def test_subtract_float(self):
        self.assertEqual(calculator("3.0 - 1.5"), 1.5)

    def test_multiply_simple(self):
        self.assertEqual(calculator("3 * 4"), 12)

    def test_multiply_float(self):
        self.assertEqual(calculator("2.5 * 2"), 5.0)

    def test_divide_simple(self):
        self.assertEqual(calculator("10 / 2"), 5.0)

    def test_divide_float(self):
        self.assertEqual(calculator("7 / 2"), 3.5)

    def test_spaces_around_operator(self):
        self.assertEqual(calculator(" 2 + 3 "), 5)
        self.assertEqual(calculator("5*2"), 10)
        self.assertEqual(calculator("  6  /  3  "), 2.0)

    def test_invalid_format_no_operator(self):
        with self.assertRaisesRegex(ValueError, "Invalid expression format. Expected 'number operator number'."):
            calculator("2 3")

    def test_invalid_format_too_many_operators(self):
        # The regex is designed to match only one operator.
        # An expression like "2 + 3 + 4" will not match the ^...$ structure.
        with self.assertRaisesRegex(ValueError, "Invalid expression format. Expected 'number operator number'."):
            calculator("2 + 3 + 4")

    def test_invalid_operands_trigger_format_error_due_to_regex(self):
        # The current regex will fail to match if operands are not numeric,
        # thus raising the "Invalid expression format" error before attempting float conversion.
        with self.assertRaisesRegex(ValueError, "Invalid expression format. Expected 'number operator number'."):
            calculator("abc + 3")
        with self.assertRaisesRegex(ValueError, "Invalid expression format. Expected 'number operator number'."):
            calculator("3 + def")
        with self.assertRaisesRegex(ValueError, "Invalid expression format. Expected 'number operator number'."):
            calculator("3.0 + def")
        # If the regex were more permissive and allowed non-numeric groups that then failed float conversion,
        # a different error ("Invalid number format.") would be tested separately.

    def test_unsupported_operator(self):
        # The regex should prevent this, but if it somehow passed, the internal logic should catch it.
        # However, the regex is quite specific. This test primarily ensures the regex doesn't match an unsupported operator.
        with self.assertRaisesRegex(ValueError, "Invalid expression format. Expected 'number operator number'."):
            calculator("2 ^ 3")
        # If we want to test the internal operator check (post-regex), the regex would need to be more permissive.
        # For now, the regex handles this by not matching.

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            calculator("10 / 0")
        with self.assertRaises(ZeroDivisionError):
            calculator("10 / 0.0")
        with self.assertRaises(ZeroDivisionError):
            calculator("0 / 0") # Also a ZeroDivisionError

    def test_negative_numbers(self):
        self.assertEqual(calculator("-2 + 3"), 1)
        self.assertEqual(calculator("5 + -2"), 3)
        self.assertEqual(calculator("-5 - -2"), -3)
        self.assertEqual(calculator("5 * -2"), -10)
        self.assertEqual(calculator("-10 / -2"), 5.0)
        self.assertEqual(calculator("10 / -2.0"), -5.0)

    def test_numbers_with_leading_plus(self): # Optional, depends on regex strictness
        self.assertEqual(calculator("+2 + 3"), 5)
        self.assertEqual(calculator("5 + +3"), 8)


# To keep the old structure if other tests are eventually added to TestTools:
class TestTools(unittest.TestCase):
    def test_calculator_tool(self):
        # This can be removed if TestCalculator covers everything needed for the calculator.
        # Or it could be a higher-level integration test if AiHelper used calculator.
        pass

    def test_weather_tool(self):
        pass

    def test_pdf_reader_tool(self):
        pass

if __name__ == '__main__':
    unittest.main()
