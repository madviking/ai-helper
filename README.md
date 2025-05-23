### Ai Helper
This is a simple ai helper to connect with openrouter, google, openai and openrouter. Input and output are always a pydantic model. Supports text, images and files and function calls.

### Activating the environment
Simply run install.sh and then do source venv/bin/activate

#### LLM TASK 
Please implement the full functionality as outlined in this document. Success is determined by the successful completion of all tests and the ability to run the example.py file without errors.

#### Specific implementation considerations
- When loading PydanticModel with results, in most cases invididual fields that don't validate should be just discarded
- We should add information about how many percent of the model fields are filled

#### Guidelines for implementation
- No function should be longer than 200 lines.
- No class should be longer than 700 lines.
- Feel free to create new files to make things more modular.
- .env contains credentisals. env-example is provided.
- ALWAYS write tests before implementing. TDD!
- ALWAYS stop for approval after creating the tests. 
- ALWAYS run tests after making changes.
- ALWAYS rely on providers for getting and modifying the LLM's, Configs, and Pydantic models.
- PATHS should always be coming from the utils, never hard coded.
- When changing any methods, ALWAYS search for usages elsewhere.
- To setup the project, run install.sh and then source venv/bin/activate

### Notes about this branch
Model:        Grok-3
Total cost:   
Wall time:    

#### Development flow with Cline:
- **[prompt]** "please read README.md and proceed with the full implementation of this project.
- **[observation]** Started by implementing all the tests and part of the functionality. Tests are mostly bs.
  - **[cost]** $2.5
- **[prompt]** Please read README.md and continue with the full implementation of this project. Target is to be able to run "python -m example" without errors in addition to tests.
