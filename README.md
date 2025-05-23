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
Total cost:   $28.4
Wall time:    Some hours

#### Development flow with Cline:
- **[prompt]** "please read README.md and proceed with the full implementation of this project.
- **[observation]** Stopped to ask feedback on the first test file. Decided to steer a little due to seeing problems where other LLM's ended up confusing too much...
- **[prompt]** "Let's implement all tests initially without mocking. So treat these as a combination of unit tests and integration tests..
- **[prompt]** "let's not add pypdf at all. We want to send pdf files as files to llm's.
- **[observation]** Individual prompts with Google are very small. Looks like it is much better at either caching or splitting the work. Whereas Claude was using a huge amount of tokens, Gemini is keeping at $0.05 per prompt or so.
- **[observation]** I'm super impressed how far we've got with just the base prompt and couple prompts mid-fly. Still without restarting.
- **[prompt]** "google-generativeai has been deprecated and the latest library we use is google-genai. See this: https://github.com/googleapis/python-genai.
- **[observation]** Unfortunately Google didn't handle situation well with info missing on its OWN latest sdk. 
  - **[cost]** $11.5
- **[prompt]** Please read the README.md. Implementation is not finished. Pay special attention into using PydanticAI. Now the adapters are parsing json etc. whereas they should be leveraging on PydanticAI.
- **[prompt]** Needed several prompts, including pasting pydanticai examples and their own toolkit's initialization instructions.
  - **[cost]** $1.8

- **[prompt]** please read the README.md. Let's continue with the implementation. 
1) Pytests should run without errors
2) python -m example --- this should display example of ACTUAL api call to all providers + tool calling + file processing
3) cost tracker should download the model pricing from openrouter and use that in the calculations
4) cost tracker should save the files with totals, llm model specific spend and pydantic model specific spend
5) adapters should return a pydantic model. I suggest we create a base model where we put cost tracking as a separate pydantic submodel. If all pydantic models extend from this base class, we can have all runtime info available through the models. 
6) Fill calculation etc. could be done there also
- **[prompt]** stopped couple of times to ask for examples from PydanticAI documentation and genai toolkit. Provided raw copy+paste of the examples.
- **[observation]** Gemini's reasoning seems more sensible than Grok-3 and Opus-4. It also asks the most relevant questions and keeps going longer without needing input. It was also the only model that I saw checking functionality from the actual packages inside venv.
  - **[cost]** $15.1
