"""Constrained-decoding engine that produces schema-valid function calls.

Drives token-by-token generation so the model can only emit text that
is both valid JSON and compliant with the schema in
``functions_definition.json``.

At each step the decoder fetches logits, asks :class:`FunctionTrie`
(and the value states) which tokens are legal, sets the logits of all
other tokens to ``-inf``, and picks the highest-scoring legal token.
The loop repeats until a complete function call has been produced.
"""
try:
    import sys
    from typing import Any
    from .vocabulary import IDTokenTransformation
    from llm_sdk import Small_LLM_Model
    from .models import FunctionDefinition
except ModuleNotFoundError:
    print("Download the dependencies by running make run or make install")
    sys.exit(1)
except KeyboardInterrupt:
    print("BRUHHHHHHHHH, I saw you brada !!!")
    sys.exit(130)


class TrieNode:
    """A single node in a :class:`FunctionTrie`.

    Attributes:
        children: Maps a token ID to the next node.
        is_end: True when this node terminates a complete token sequence.
    """
    def __init__(self) -> None:
        """A single node in a :class:`FunctionTrie`.

        Attributes:
            children: Maps a token ID to the next :class:`TrieNode`.
            is_end: True when this node terminates a complete token sequence
                (i.e. a full function name has been generated).
        """
        self.children: dict[int, "TrieNode"] = {}
        self.is_end: bool = False


class FunctionTrie:
    """Prefix tree of tokenised function names used for constrained decoding.

    Each path from the root to a node marked ``is_end`` spells the token
    sequence of exactly one registered function name. During generation
    the decoder walks this trie so the LLM can only emit tokens that keep
    the partially generated name on a valid path.

    Attributes:
        root: The root node of the trie.
    """

    def __init__(self) -> None:
        """Initialise an empty trie with a single root node."""
        self.root = TrieNode()

    def insert(self, token_ids: list[int]) -> None:
        """Insert a token sequence into the trie.

        Args:
            token_ids: Token IDs spelling a function name.
        """
        current = self.root
        for token_id in token_ids:
            if token_id not in current.children:
                current.children[token_id] = TrieNode()
            current = current.children[token_id]
        current.is_end = True

    def get_node(self, token_ids: list[int]) -> TrieNode | None:
        """Return the node reached by following ``token_ids`` from the root.

            Used to verify if the token_ids is a proper function name,
            by checking self.is_end is True.

        Args:
            token_ids: The prefix to look up.

        Returns:
            The matching node, or None if the prefix is not in the trie.
        """
        current = self.root
        for token_id in token_ids:
            if token_id not in current.children:
                return None
            current = current.children[token_id]
        return current

    def get_allowed_tokens(self, token_ids: list[int]) -> list[int]:
        """Return token IDs that can legally extend ``token_ids``.

        Args:
            token_ids: The prefix generated so far.

        Returns:
            The list of token IDs reachable from the current node.
        """
        node = self.get_node(token_ids)
        if node is None:
            return []
        return list(node.children.keys())


class DecoderState:
    """Base class for one step of the argument-generation plan.

    Subclasses represent either forced literal text, a value to be
    generated under type constraints, or the end of the plan.

    Attributes:
        state_type: Short string tag identifying the concrete subclass.
    """

    state_type: str = "base"

    def __repr__(self) -> str:
        """Return a debug representation of the state."""
        return f"<{self.__class__.__name__}>"


class ForcedTextState(DecoderState):
    """Initialise the state.

    Args:
        text: The literal string that must be appended verbatim to the
            running token stream.
    """

    state_type = "forced_text"

    def __init__(self, text: str) -> None:
        """Initialise the state.

        Args:
            text: The literal string that must be appended verbatim.
        """
        self.text: str = text
        self.token_ids: list[int] | None = None

    def __repr__(self) -> str:
        """Return a debug representation of the state."""
        return f"<ForcedTextState text={self.text!r}>"


class ValueState(DecoderState):
    """A parameter value the LLM must generate under type constraints.

    Attributes:
        parameter_name: JSON key (e.g. "a").
        value_type: "number" | "string" | "boolean".
        current_text: Value text generated so far.
    """

    state_type = "value"

    def __init__(self, parameter_name: str, value_type: str) -> None:
        """Initialise the state.

        Args:
            parameter_name: JSON key of the parameter (e.g. ``"a"``).
            value_type: One of ``"number"``, ``"string"``, ``"boolean"``.
        """
        self.parameter_name: str = parameter_name
        self.value_type: str = value_type
        self.current_text: str = ""

    def __repr__(self) -> str:
        """Return a debug representation of the state."""
        return (
            f"<ValueState param={self.parameter_name!r} "
            f"type={self.value_type!r} current={self.current_text!r}>"
        )


class DoneState(DecoderState):
    """Marker state signalling the end of a decoding plan.

    The decoder stops iterating as soon as it encounters this state.
    """
    state_type = "done"


def build_states(function_def: FunctionDefinition) -> list[DecoderState]:
    """Turn a function definition into an ordered decoding plan.

    Example for `fn_add_numbers(a: number, b: number)`:
        ForcedTextState('{"a":')
        ValueState("a", "number")
        ForcedTextState(', "b":')
        ValueState("b", "number")
        ForcedTextState('}')
        DoneState()

    Args:
        function_def: Validated function definition.

    Returns:
        The plan, ready to be executed by `run_states()`.
    """
    states: list[DecoderState] = []
    params = list(function_def.parameters.items())

    if not params:
        states.append(ForcedTextState("{}"))
        states.append(DoneState())
        return states

    states.append(ForcedTextState("{"))
    for index, (name, param) in enumerate(params):
        prefix = f'"{name}":' if index == 0 else f', "{name}":'
        states.append(ForcedTextState(prefix))
        value_type = param.param_types.value
        if value_type == "integer":
            value_type = "integer"
        if value_type == "string":
            states.append(ForcedTextState('"'))
        states.append(ValueState(name, value_type))
    states.append(ForcedTextState("}"))
    states.append(DoneState())
    return states


def append_forced_text(
    state: ForcedTextState,
    instance: Small_LLM_Model,
    current_tokens: list[int],
) -> None:
    """Encode `state.text` and append its token IDs to the context.

    Args:
        state: The ForcedTextState to apply.
        instance: LLM SDK (for `encode`).
        current_tokens: Running token list, modified in place.
    """
    if state.token_ids is None:
        encoded = instance.encode(state.text)
        state.token_ids = encoded.flatten().int().tolist()
    current_tokens.extend(state.token_ids)


def legal_tokens_for_value(
    state: ValueState,
    vocab: IDTokenTransformation,
    terminators: list[int],
) -> list[int]:
    """Tokens that may legally continue the current value.

    Combines vocabulary tokens valid for the value's type with
    terminator tokens — but only when the value is non-empty, so
    we never produce `{"a":, ...}`.

    Args:
        state: The current ValueState.
        vocab: Vocabulary helper.
        terminators: Token IDs that end a value.

    Returns:
        Allowed token IDs.
    """
    allowed = list(
        vocab.tokens_for_type(state.value_type, state.current_text)
    )
    if state.current_text:
        allowed.extend(terminators)
    return allowed


def run_value_generation(
    state: ValueState,
    instance: Small_LLM_Model,
    vocab: IDTokenTransformation,
    current_tokens: list[int],
    terminators: list[int],
    max_tokens: int = 32,
) -> None:
    """Generate one parameter value token-by-token under type constraints.

    Loop:
        1. Get logits for the current context.
        2. Ask the vocabulary which tokens are legal.
        3. Add terminators if the value is non-empty.
        4. Mask logits.
        5. Pick the highest-logit legal token.
        6. Append it; grow current_text.
        7. Stop on terminator or after `max_tokens`.

    Args:
        state: ValueState, mutated in place.
        instance: LLM SDK.
        vocab: Vocabulary helper.
        current_tokens: Running token list, modified in place.
        terminators: Terminator token IDs.
        max_tokens: Safety cap against infinite loops.
    """
    terminator_set = set(terminators)

    for _ in range(max_tokens):
        logits_list = instance.get_logits_from_input_ids(current_tokens)

        allowed = legal_tokens_for_value(state, vocab, terminators)
        if len(state.current_text) > 64:
            allowed = terminators
        if not allowed:
            return

        masked = mask_logits(logits_list, allowed)
        selected = masked.index(max(masked))

        if selected in terminator_set and state.current_text:
            return

        current_tokens.append(selected)
        state.current_text += instance.decode([selected])


def _coerce(text: str, value_type: str) -> Any:
    """Convert generated value text into its Python representation.

    Args:
        text: Value text produced by the LLM.
        value_type: "number" | "string" | "boolean".

    Returns:
        Python value (float / str / bool).
    """
    if value_type == "number":
        return float(text)
    if value_type == "integer":
        return int(text)
    if value_type == "boolean":
        return text.strip().lower() == "true"
    return text.strip()


def run_states(
    states: list[DecoderState],
    instance: Small_LLM_Model,
    vocab: IDTokenTransformation,
    current_tokens: list[int],
) -> dict[str, Any]:
    """Execute a decoding plan and return the parsed parameters.

    Args:
        states: Plan produced by `build_states()`.
        instance: LLM SDK.
        vocab: Vocabulary helper.
        current_tokens: Running token list, modified in place.

    Returns:
        Dict mapping parameter names to Python values.
    """
    parameters: dict[str, Any] = {}

    comma_ids = vocab.tokens_starting_with(",")
    brace_ids = vocab.tokens_starting_with("}")
    quote_ids = vocab.tokens_starting_with('"')
    terminators = comma_ids + brace_ids + quote_ids

    for state in states:
        if isinstance(state, ForcedTextState):
            append_forced_text(state, instance, current_tokens)

        elif isinstance(state, ValueState):
            run_value_generation(
                state, instance, vocab, current_tokens, terminators
            )
            parameters[state.parameter_name] = _coerce(
                state.current_text, state.value_type
            )

        elif isinstance(state, DoneState):
            break

    return parameters


def mask_logits(logits: list[float], allowed_tokens: list[int]) -> list[float]:
    """Replace logits of illegal tokens with negative infinity.

    Args:
        logits: Raw logits, one per vocabulary entry.
        allowed_tokens: Token IDs that may legally be selected next.

    Returns:
        A new list where entries whose index is not in ``allowed_tokens``
        are set to ``float("-inf")``; legal entries are unchanged.
    """
    allowed = set(allowed_tokens)
    masked_logits = [value if token_id in allowed else float("-inf")
                     for token_id, value in enumerate(logits)]
    return masked_logits
