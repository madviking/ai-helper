### Ai Helper

This is a simple ai helper to connect with openrouter, google, openai and openrouter. Input and output is always a pydantic model. 

### Activating the environment

Simply run install.sh and then do source venv/bin/activate

#### LLM TASK 
Please implement the full functionality. 

#### Specific implementation considerations
- When loading PydanticModel with results, in most cases invididual fields that don't validate should be just discarded
- We should add information about how many percent of the model fields are filled


#### Guidelines for implementation
- No function should be longer than 200 lines.
- No class should be longer than 700 lines.
- Feel free to create new files to make things more modular.
- ALWAYS write tests before implementing. TDD!
- ALWAYS stop for approval after creating the tests. 
- ALWAYS run tests after making changes.
- PATHS should always be coming from the utils, never hard coded.
- Use the provided venv information. **Don't install packages or try to modify venv**. Always activate it to run any commands.
- When changing any methods, ALWAYS search for usages elsewhere.


### Task
Please do a simple implementation with the following structure:

py_models
    pd_reader_model.py
    weather_model.py
tests
    test_integrations.py
    files/
        inputs/
        outputs/
        test.pdf
        test.png

src/
    ai_helper.py                --- main class
    tools.py                    --- two example tools
    cost_tracker.py             --- each run's costs are saved here
    adapters/
        anthropic.py                
        google.py
        openai.py
        openrouter.py
    
example.py

