# AI Helper Project - Handover Document

## Project Overview
This is a Python library that provides a unified interface for interacting with multiple LLM providers (OpenAI, Anthropic, Google, OpenRouter) with support for:
- Pydantic model validation for structured outputs
- Tool/function calling
- File input (text and images)
- Cost tracking
- Multiple adapters for different providers

## Current Status

### ✅ Completed Components

1. **Core Implementation**
   - `src/ai_helper.py` - Main AIHelper class with adapter pattern
   - `src/cost_tracker.py` - Token usage and cost tracking
   - `src/tools.py` - Tool preparation utilities
   - All adapter implementations (OpenAI, Anthropic, Google, OpenRouter)

2. **Adapter Implementations**
   - `src/adapters/base_adapter.py` - Abstract base class
   - `src/adapters/openai.py` - OpenAI adapter with vision and tools support
   - `src/adapters/anthropic.py` - Anthropic adapter with tools support
   - `src/adapters/google.py` - Google Gemini adapter with vision support
   - `src/adapters/openrouter.py` - OpenRouter adapter with model mapping

3. **Example Models**
   - `py_models/weather_model.py` - Weather data model
   - `py_models/pd_reader_model.py` - PDF reader model
   - `py_models/general_example_model.py` - General example model

### 🔧 In Progress - Test Fixes

The main task currently in progress is fixing the adapter tests. The tests are failing because they're trying to create real API clients even though we're mocking them.

#### Issue Details:
1. **Problem**: Tests are creating adapters in `setUp()` before mocks are applied
2. **Solution Started**: Adding mocks to `setUp()` methods for each test class
3. **Status**: 
   - ✅ Fixed `TestOpenAIAdapter.setUp()`
   - ✅ Fixed `TestAnthropicAdapter.setUp()`
   - ❌ Still need to fix `TestGoogleAdapter` and `TestOpenRouterAdapter`

#### Current Test Failures:
```
FAILED tests/test_adapters.py::TestOpenAIAdapter::test_openai_adapter_text_only
FAILED tests/test_adapters.py::TestOpenAIAdapter::test_openai_adapter_with_image
FAILED tests/test_adapters.py::TestOpenAIAdapter::test_openai_adapter_with_tools
FAILED tests/test_adapters.py::TestAnthropicAdapter::test_anthropic_adapter_text_only
FAILED tests/test_adapters.py::TestAnthropicAdapter::test_anthropic_adapter_with_tools
FAILED tests/test_adapters.py::TestGoogleAdapter::test_google_adapter_text_only
FAILED tests/test_adapters.py::TestGoogleAdapter::test_google_adapter_with_image
FAILED tests/test_adapters.py::TestOpenRouterAdapter::test_openrouter_adapter_headers
FAILED tests/test_adapters.py::TestOpenRouterAdapter::test_openrouter_adapter_model_mapping
```

## Next Steps

### 1. Fix Remaining Adapter Tests
- Fix `TestGoogleAdapter.setUp()` to properly mock the Google client
- Fix `TestOpenRouterAdapter.setUp()` to properly mock the OpenAI client
- Update test methods to remove duplicate mocking since it will be in setUp()
- Ensure all tests pass

### 2. Fix Google Adapter Import Issue
The Google adapter tests are failing with:
```
AttributeError: module 'google' has no attribute 'genai'
```
This suggests the Google Generative AI library might not be properly installed or the import path has changed.

### 3. Complete Test Suite
Once adapter tests are fixed, run the full test suite:
```bash
python3 -m pytest tests/ -v
```

### 4. Integration Testing
The integration tests (`test_integrations.py`) should be reviewed to ensure they work with the fixed adapters.

## Technical Notes

### Test Mocking Pattern
The correct pattern for mocking adapters in tests is:

```python
class TestAdapterClass(unittest.TestCase):
    @patch('module.ClientClass')
    @patch.dict(os.environ, {'API_KEY_NAME': 'test_key'})
    def setUp(self, mock_client_class):
        """Set up test fixtures"""
        # Mock the client
        self.mock_client = MagicMock()
        mock_client_class.return_value = self.mock_client
        self.adapter = AdapterClass()
```

### Environment Variables
The project expects these environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_KEY`

### Dependencies
All dependencies are listed in `requirements.txt` and the project uses a virtual environment.

## File Structure
```
ai_helper/
├── src/
│   ├── adapters/
│   │   ├── base_adapter.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── google.py
│   │   └── openrouter.py
│   ├── ai_helper.py
│   ├── cost_tracker.py
│   └── tools.py
├── tests/
│   ├── test_adapters.py (currently being fixed)
│   ├── test_ai_helper.py
│   ├── test_cost_tracker.py
│   ├── test_integrations.py
│   ├── test_py_models.py
│   └── test_tools.py
├── py_models/
│   ├── weather_model.py
│   ├── pd_reader_model.py
│   └── general_example_model.py
└── example.py
```

## Commands to Resume Work

1. Activate virtual environment (already active)
2. Run specific test file:
   ```bash
   python3 -m pytest tests/test_adapters.py -v
   ```
3. Run all tests:
   ```bash
   python3 -m pytest tests/ -v
   ```

## Recent Changes
- Modified adapter test classes to properly mock API clients in setUp()
- Fixed import issues in test files
- Updated test structure to prevent real API calls during testing

## Known Issues
1. Google Generative AI import issue needs investigation
2. Some test methods have duplicate mocking that should be removed
3. OpenRouter adapter tests need proper client mocking in setUp()

## Recommendations
1. Focus on fixing the remaining adapter test setUp() methods first
2. Investigate and fix the Google Generative AI import issue
3. Clean up duplicate mocking in test methods
4. Run full test suite to ensure everything works together
5. Consider adding more integration tests for real-world scenarios
