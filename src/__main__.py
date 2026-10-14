"""Entry point for the constrained-decoding function-calling tool.

Orchestrates the complete pipeline:
load input files, validate them with Pydantic, tokenise the prompt,
generate the function name via the trie, generate each argument via
the value states, and write the final JSON result.
"""

try:
    import sys
    from .function_registry import FunctionRegistration
    from .models import FunctionDefinition, FunctionCallTest, FunctionCall
    from .vocabulary import IDTokenTransformation
    from .decoder import FunctionTrie, mask_logits, build_states, run_states
    from llm_sdk import Small_LLM_Model
    from .ui import (
                    print_banner,
                    print_section,
                    print_prompt,
                    print_result,
                    print_error,
                    print_summary,
                )
    from typing import Any
    import json
    import torch
    import os
    import argparse
except ModuleNotFoundError:
    print("Download the dependencies by running make run or make install")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input/output file paths.

    Returns:
        Namespace with functions_definition, input, and output attributes.
    """
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description="Constrained-decoding function-calling tool.",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the natural-language prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path where the results JSON will be written.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all decoration; print only JSON to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full function-calling pipeline.

    Returns:
        None.
    """
    args = parse_args()

    print_banner(version="0.1.0")
    print_section("Loading input files")

    try:
        with open(
                args.functions_definition,
                "r") as tools_collection:
            file_content = json.load(tools_collection)
        result_ft_dev_val = [FunctionDefinition.model_validate(function)
                             for function in file_content]
    except Exception:
        if not args.quiet:
            print_error("There is an error in the functions_"
                        "definition.json file.\n"
                        "Either: corrupted JSON file or missing file"
                        )
        else:
            print("There is an error in the functions_"
                  "definition.json file.\n"
                  "Either:\n"
                  "- Corrupted JSON file\n"
                  "- Missing file\n")
        sys.exit(1)

    try:
        with open(args.input, "r") as prompt:
            prompt_command = json.load(prompt)
        result_ftcall_test = [FunctionCallTest.model_validate(prompting)
                              for prompting in prompt_command]
    except Exception:
        if not args.quiet:
            print_error("There is an error in the function_"
                        "calling_tests.json file.\n"
                        "Either: corrupted JSON file or missing file"
                        )
        else:
            print("There is an error in the functions_"
                  "calling_tests.json file.\n"
                  "Either:\n"
                  "- Corrupted JSON file\n"
                  "- Missing file\n")
        sys.exit(1)

    print(f"        Loaded {len(file_content)} function(s) "
          f"and {len(prompt_command)} prompt(s).")

    def dict_to_str(prompt: list[dict[str, str]]) -> str:
        """Combine function definitions and prompts into a single string.

        Args:
            prompt: Parsed contents of ``function_calling_tests.json``.

        Returns:
            A single string listing every function declaration followed by
            every prompt, ready to be tokenised and fed to the LLM.
        """
        text = ""
        reg = FunctionRegistration(file_content)

        class Registration:
            """Render function definitions into a flat textual description.

            Attributes:
                func_registration: Registry providing per-function lookups.
                function_name: Names of all registered functions.
                text: Accumulated textual description.
            """
            def __init__(self, reg: FunctionRegistration = reg) -> None:
                """Initialise the renderer.

                Args:
                    reg: Registry of function definitions to render.
                """
                self.func_registration = reg
                self.function_name = self.func_registration.get_all_functions()
                self.text: str = ""

            def _name(self, info: str) -> Any:
                """Return the name of the function identified by ``info``.

                Args:
                    info: The function name.

                Returns:
                    The same name, fetched through the registry.
                """
                return self.func_registration.get_function(info).name

            def _description(self, info: str) -> Any:
                """Return description of the function identified by ``info``.

                Args:
                    info: The function name.

                Returns:
                    The function's human-readable description.
                """
                return self.func_registration.get_function(info).description

            def _parameter(self, info: str) -> str:
                """Render the parameter block of one function.

                Args:
                    info: The function name.

                Returns:
                    A multi-line string with one ``- name: (type)`` entry per
                    parameter.
                """
                _parameter: dict[str, dict[str, str]] = \
                    self.func_registration.get_function(info).parameters
                param_val: str = ""
                for key, value in _parameter.items():
                    for v in value.values():
                        param_val += f"- {key}: ({v})\n"
                return param_val

            def _returns(self, info: str) -> str:
                """Render the return type of one function.

                Args:
                    info: The function name.

                Returns:
                    A string of the form ``Returns: (type)``.
                """
                _returns: dict[str, Any] = \
                    self.func_registration.get_function(info).returns
                ret_val: str = ""
                for value in _returns.values():
                    ret_val = f"Returns: ({value})"
                return ret_val

            def pilot(self) -> str:
                """Render every registered function into ``self.text``.

                Returns:
                    The accumulated multi-function description.
                """
                for function in self.function_name:
                    self.text += f"{self._name(function)}: " \
                        f"{self._description(function)}\n" \
                        f"Parameter(s):\n{self._parameter(function)}" \
                        f"{self._returns(function)}\n\n"
                return self.text
        declaration = Registration()
        text = declaration.pilot()

        for prompt_list in prompt:
            for value in prompt_list.values():
                text += f"Prompt: {value}\n"

        return text
    FUNCTIONS_BLOCK: str = dict_to_str([])

    if not args.quiet:
        print_section("Loading model")
    instance = Small_LLM_Model()
    instance_vocab = IDTokenTransformation(instance)

    trie = FunctionTrie()
    func_registration = FunctionRegistration(file_content)
    func_list = func_registration.get_all_functions()

    for function_name in func_list:
        token_ids = instance.encode(function_name).flatten().int().tolist()
        trie.insert(token_ids)

    calls: list[FunctionCall] = []

    if not args.quiet:
        print_section("Generating function calls")

    total = len(result_ftcall_test)

    for index, test in enumerate(result_ftcall_test, start=1):

        if not args.quiet:
            print_prompt(index, total, test.prompt)

        TEXT: str = FUNCTIONS_BLOCK + f"Prompt: {test.prompt}\n"

        input_token_ID = instance.encode(TEXT)
        current_tokens = input_token_ID.flatten().int().tolist()

        generated_function_tokens: list[int] = []
        while True:
            logits = instance.get_logits_from_input_ids(current_tokens)
            allowed_tokens = trie.get_allowed_tokens(
                generated_function_tokens
            )
            masked_logits = mask_logits(logits, allowed_tokens)
            selected_token_id = masked_logits.index(max(masked_logits))
            generated_function_tokens.append(selected_token_id)
            current_tokens.append(selected_token_id)
            node = trie.get_node(generated_function_tokens)
            if node is not None and node.is_end:
                break

        generated_function_name = instance.decode(
            torch.tensor(generated_function_tokens)
        )

        func_def = next(
            (f for f in result_ft_dev_val
             if f.name == generated_function_name),
            None,
        )
        if func_def is None:
            if not args.quiet:
                print_error(
                    f"no definition for {generated_function_name!r}"
                )
            continue

        try:
            states = build_states(func_def)
            params = run_states(
                states, instance, instance_vocab, current_tokens
            )
        except ValueError as e:
            print_error(f"token-level error: {e}")
            sys.exit(1)

        call = FunctionCall(
            prompt=test.prompt,
            name=generated_function_name,
            parameters=params,
        )
        calls.append(call)

        if not args.quiet:
            print_result(generated_function_name, params)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as result:
        result.write(
            "[\n"
            + ",\n".join(c.model_dump_json(indent=2) for c in calls)
            + "\n]\n"
        )

    if args.quiet:
        print(
            "[\n"
            + ",\n".join(c.model_dump_json(indent=2) for c in calls)
            + "\n]\n"
        )
    else:
        print_summary(len(calls), args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("BRUHHHHHHHHH, I saw you brada !!!")
        sys.exit(130)
