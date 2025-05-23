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
Model:        Gemini-2.5-preview
Total cost:   
Wall time:    

#### Development flow with Cline:
- **[prompt]** "please read README.md and proceed with the full implementation of this project.
- **[observation]** Stopped to ask feedback on the first test file. Decided to steer a little due to seeing problems where other LLM's ended up confusing too much...
- **[prompt]** "Let's implement all tests initially without mocking. So treat these as a combination of unit tests and integration tests..
- **[prompt]** "let's not add pypdf at all. We want to send pdf files as files to llm's.
- **[observation]** Individual prompts with Google are very small. Looks like it is much better at either caching or splitting the work. Whereas Claude was using a huge amount of tokens, Gemini is keeping at $0.05 per prompt or so.
- **[observation]** I'm super impressed how far we've got with just the base prompt and couple prompts mid-fly. Still without restarting.
- **[prompt]** "google-generativeai has been deprecated and the latest library we use is google-genai. See this: https://github.com/googleapis/python-genai.
  - **[cost]** $11.5


