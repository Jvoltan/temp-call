*This project has been created as part of the 42 curriculum by jraveloa.*

# Call_Me_Maybe
![Call_Me_Maybe general visualisation](image/call_me_maybe_preview.png)
## Description

Small language models are bad at writing JSON. Often the output is broken.

This project turns a normal sentence into a **function call**. It does not answer the question. It says which function to use and with which values.

Example: for `"What is the sum of 2 and 3?"` the program gives:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

It uses the model `Qwen/Qwen3-0.6B` and **constrained decoding**: at each step, the program blocks every token that would break the format. So the JSON is always valid, even with a very small model.

## Instructions

**You need:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/), and internet on the first run (to download the model).

```bash
make install    # installs everything (uv sync)
make run        # runs the program
```

The program reads two files in `data/input/`:

- `functions_definition.json`: the functions it can call
- `function_calling_tests.json`: the sentences to convert

It handles every sentence and writes all the results to `data/output/function_calling_results.json`.

### Options

You can choose other files with `uv run python -m src [options]`:

| Option | Default | Meaning |
|---|---|---|
| `--functions_definition <file>` | `data/input/functions_definition.json` | The functions it can call |
| `--input <file>` | `data/input/function_calling_tests.json` | The sentences to convert |
| `--output <file>` | `data/output/function_calling_results.json` | Where the results are written |
| `--quiet` | off | No decoration: print only the JSON on the screen |

Example:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

If an input file is missing or is not valid JSON, the program prints a clear error message and stops with exit code 1.

### What you see in the terminal

While it runs, the program shows:

- a small banner at the start;
- a title for each step (loading files, loading the model, generating);
- one line per sentence, like `[2/5] Greet shrek`, and under it the function it chose;
- a short summary at the end with the output path.

Colors are used only if the terminal supports them. This display is all in `src/ui.py`. It only prints on the screen (stdout or stderr) and never writes in the JSON file, so the output file stays clean.

### Makefile commands

| Command | What it does |
|---|---|
| `make install` | Install the dependencies |
| `make run` | Run the program |
| `make debug` | Run with the Python debugger (`pdb`) |
| `make clean` | Delete `__pycache__` and `.mypy_cache` |
| `make lint` | Run `flake8` and `mypy` with the subject's flags |
| `make lint-strict` | Run `flake8` and `mypy --strict` |
| `make docstr` | Check the docstrings (Google style) |

The Makefile saves the model cache in `/sgoinfre/<your login>/...`. If this folder does not exist on your computer, change `HF_HOME_DIR` and `UV_CACHE_PATH` at the top of the Makefile.

**Files:**

```
src/__main__.py          main program
src/models.py            Pydantic models (input and output)
src/function_registry.py finds a function by its name
src/vocabulary.py        token lists (numbers, strings, ...)
src/decoder.py           trie, decoding steps, logit masking
src/ui.py                terminal display only (banner, colors, progress)
llm_sdk/                 the provided SDK
```

## Algorithm explanation

A language model writes **one token at a time**. For each step it gives a score (a *logit*) to every possible token. Normally we take the best one. With constrained decoding we do this instead:

1. Get the scores from the model.
2. Find the tokens that are **allowed** at this exact point.
3. Set the score of all other tokens to `-inf` (`mask_logits`).
4. Take the best token that is left.
5. Add it to the text and start again.

There are two phases.

**Phase 1: choose the function name.** Every function name is cut into tokens and stored in a **trie** (a tree of tokens). At each step, only the tokens that continue a real function name are allowed. The model chooses the path, but it can only follow a real name. We stop when the name is complete.

**Phase 2: write the arguments.** From the chosen function, the program builds a plan. For `fn_greet(name: string)`:

```
{          fixed text
"name":    fixed text
"          fixed text
<value>    the model writes this
"          fixed text
}          fixed text
```

- **Fixed text** is added directly. The model never writes `{`, `:`, `,` or quotes. This is why the structure is always right.
- **The value** is written token by token. The allowed tokens depend on the type:
  - *number*: only tokens that can still become a number (`-`, `1.`, `1.5e-` are ok; `--1` or `abc` are not)
  - *boolean*: only tokens that can still become `true` or `false`
  - *string*: any token without a `"`
- A value ends when the model picks a token starting with `,` or `}` (only after at least one token). That token is dropped, because the fixed text after it already contains the right ending.
- For safety, a value stops after 32 tokens, or after 64 characters.

At the end, the text is converted to the right Python type (`float`, `bool` or `str`) and saved with Pydantic.

## Design decisions

- **The program writes the structure, the model only makes choices.** The model chooses the function and the values. All the rest is fixed text.
- **A trie for function names.** A name has many tokens. The trie tells us the allowed next tokens at each step.
- **Two phases.** Once we know the function, its parameters tell us exactly what to write next. This works for any list of functions.
- **Nothing is hard-coded.** No function name or prompt from the examples is written in the code.
- **Setup once, context per prompt.** The model, the vocabulary and the trie are created once. Each prompt gets its own text, so prompts do not affect each other.
- **Greedy choice.** We always take the best allowed token, so the same input gives the same output.
- **Pydantic** checks the input files and the output. Numbers are saved as floats (`2.0`), like in the subject.
- **Display is separate from the logic.** `ui.py` only prints. The decoding code and the JSON file never depend on it, and `--quiet` can switch the decoration off.
- **Only public SDK methods** are used. All code has type hints and Google-style docstrings.

## Performance analysis

- **Valid JSON:** always, because the program writes the structure.
- **Correct function name:** always a real one, because of the trie.
- **Correct choice and values:** this depends on the small model. Constrained decoding makes the output *valid*, not always *right*.
- **Speed:** the model reads the whole text again for each new token (no cache), and masking goes over about 150k scores in Python. It works, but it is not fast.

## Challenges faced

- **Tokens are not letters.** `15` can be one token or two. Qwen also marks a space with the character `Ġ`. The vocabulary cleans this before comparing.
- **Checking half-written numbers.** `-` or `1.` are not numbers yet, but we must allow them. A small function (`_is_number_prefix`) checks the start of a number step by step.
- **Knowing when a value is finished.** The end tokens (`,` and `}`) are allowed only after the value has started, so we never get an empty value.
- **Strings.** Tokens with `"` are blocked and the quotes are fixed text, so a string cannot break the JSON.
- **Keeping prompts separate.** The text is rebuilt for each prompt, and all results are written once at the end.

## Testing strategy

- **Lint:** `make lint` and `make docstr`. At the time of writing, `flake8`, `mypy` and the docstring check show no error on `src/`.
- **Parts without the model:** `_is_number_prefix` (accepts `"1."`, `"1.5e-3"`; refuses `"--1"`, `"abc"`), the trie, `build_states` and `mask_logits` can be tested alone.
- **Full run:** run `make run`, then check the output file: valid JSON, only the keys `prompt`, `name`, `parameters`, a real function name, all parameters present, and the right types.
- **Edge cases:** empty strings, big numbers, special characters, unclear prompts, and functions with several parameters.

## Example usage

```bash
make install
make run
```

Input (`data/input/function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Expected result in `data/output/function_calling_results.json`:

| Prompt | Function | Parameters |
|---|---|---|
| `What is the sum of 2 and 3?` | `fn_add_numbers` | `{"a": 2.0, "b": 3.0}` |
| `Greet shrek` | `fn_greet` | `{"name": "shrek"}` |
| `Reverse the string 'hello'` | `fn_reverse_string` | `{"s": "hello"}` |

With your own files, and with only the JSON on the screen:

```bash
uv run python -m src --input my_tests.json --output results/my_result.json
uv run python -m src --quiet
```

## Known limitations

- A string with a comma inside can be cut at the comma. Very long values are cut (32 tokens / 64 characters).
- A number can stop at an unfinished form like `-` or `1e`.
- If one function name is the beginning of another one, the trie cannot tell them apart.
- If no function matches the name, a warning is shown and the prompt is skipped.
- With `--quiet`, a few lines (the banner, the first title and the "Loaded ... function(s)" line) are still printed before the JSON.
- The `--output` path must contain a folder (for example `results/out.json`). A bare name like `out.json` fails.

## Resources

- ZeroEntropy, [*Constrained decoding: forcing LLM output to a grammar*](https://zeroentropy.dev/concepts/constrained-decoding/): constrained decoding
- [Hugging Face docs](https://huggingface.co/docs), [Pydantic docs](https://docs.pydantic.dev/), [uv docs](https://docs.astral.sh/uv/)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [PEP 8](https://peps.python.org/pep-0008/), [PEP 257](https://peps.python.org/pep-0257/), [PEP 484](https://peps.python.org/pep-0484/), [flake8](https://flake8.pycqa.org/), [mypy](https://mypy.readthedocs.io/)

### How AI was used

- **README:** Creation and formatting based on information provided to the AI.
- **Project roadmap:** Creation of the project structure and a roadmap covering the necessary steps.
- **Explanation:** Detailed breakdowns of complex components (especially tokenization).
- **Other:** Assistance with debugging and Makefile verification, including providing and explaining an optimized professional version.
