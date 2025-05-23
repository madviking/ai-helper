## Ai Helper
This is a simple ai helper to connect with openrouter, google, openai and openrouter. Input and output are always a pydantic model. Supports text, images and files and function calls.

#### Activating the environment
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
Model:        Claude Opus 4
Total cost:   
Wall time:    Too much... this thing is slow

#### Development flow with Cline:
- Started by implementing all the tests
  - **[cost]** $1.7
- Somehow got stuck and after this each call cost more than 1.5$ per shot, restarted. 
  - **[cost]** Extra cost: $6.
- **[prompt]** "please read README.md and proceed with the full implementation of this project. Tests are already implemented, so no need to check those for now. Also, install has been ran and you are currently in active venv session."
- **[observation]** seems to be very consistent with implementation steps, moving from one file to another
- **[observation]** hardcoded model info into adapters though there is a file we are pulling from openrouter (though this is not instructed, it can be inferred)
- **[observation]** some calls were hitting already 1.5$ / req
- **[observation]** forgot that we are supposed to operate inside venv
- **[observation]** missed to abstract leading to code repetition between adapters
- **[observation]** Claude hasn't tested running "python -m example" at all :(
- **[observation]** started getting expensive, asked it to create a handover document
  - **[cost]** $15.9

#### Summary of phase 1
Nothing works, money spent is at around $23.6
