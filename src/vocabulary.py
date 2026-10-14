"""Token-ID <-> vocabulary-text mapping for constrained decoding.

Loads the model vocabulary JSON once and pre-computes indexes
(numeric tokens, string-safe tokens, tokens by first character) so the
decoder can cheaply ask "which tokens are legal right now?".
"""
import sys
try:
    from llm_sdk import Small_LLM_Model
    import json
except KeyboardInterrupt:
    print("BRUHHHHHHHHH, I saw you brada !!!")
    sys.exit(130)
_NUMBER_CHARS = set("0123456789.-+eE")


class IDTokenTransformation:
    """Bidirectional mapping between token IDs and vocabulary text.

    Loads the model's vocabulary JSON once and pre-computes indexes
    used by the constrained decoder (numeric tokens, string-safe tokens,
    tokens grouped by first character, etc.).

    Attributes:
        model_instance: The LLM SDK instance providing the vocab path.
        vocab_path: Filesystem path to the vocabulary JSON.
        id_to_text: Mapping ``token_id -> token_text``.
        text_to_id: Mapping ``token_text -> token_id``.
        by_first_char: Tokens bucketed by their first non-space character.
        numeric_tokens: Tokens whose text is a valid numeric fragment.
        string_safe_tokens: Tokens whose text contains no double quote.
    """

    def __init__(self, instance: Small_LLM_Model) -> None:
        """Load the vocabulary and build all lookup indexes.

        Args:
            instance: LLM SDK used to locate the vocabulary JSON file.
        """
        self.model_instance = instance
        self.vocab_path: str = instance.get_path_to_vocab_file()
        self.id_to_text: dict[int, str] = {}
        self.text_to_id: dict[str, int] = {}
        self.by_first_char: dict[str, list[int]] = {}
        self.numeric_tokens: list[int] = []
        self.string_safe_tokens: list[int] = []
        try:
            self._load_vocabulary()
        except (OSError, json.JSONDecodeError):
            print("Error: either the vocabulary cannot be opened or file is "
                  "not valid JSON")
        self._build_indexes()

    def _load_vocabulary(self) -> None:
        """Load the vocabulary JSON and populate the lookup tables.

        Raises:
            OSError: If the vocabulary file cannot be opened.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            raw: dict[str, int] = json.load(f)

        for token_text, token_id in raw.items():
            normalized = token_text.replace("\u0120", " ")
            self.id_to_text[int(token_id)] = normalized
            if normalized not in self.text_to_id:
                self.text_to_id[normalized] = int(token_id)

    def all_tokens(self) -> dict[int, str]:
        """Return all vocabulary tokens indexed by token ID.

        Returns:
            A dict mapping ``token_id -> token_text``.
        """
        return self.id_to_text

    def _stripped(self, token_id: int) -> str:
        """Return token text with leading whitespace removed.

        Qwen encodes a leading space as the special character ``Ġ``; this
        helper normalises it back to a real space before stripping.

        Args:
            token_id: The token whose text should be normalised.

        Returns:
            The token text without leading whitespace.
        """
        return self.all_tokens()[token_id].replace("\u0120", " ").lstrip()

    def _build_indexes(self) -> None:
        """Pre-compute helper indexes from ``id_to_text``.

        Populates ``by_first_char``, ``numric_tokens`` and
        ``string_safe_tokens`` so that later lookups are O(1).
        """
        for tid in self.all_tokens().keys():
            stripped = self._stripped(tid)
            if stripped:
                self.by_first_char.setdefault(stripped[0], []).append(tid)
                if all(c in _NUMBER_CHARS for c in stripped):
                    self.numeric_tokens.append(tid)

            if '"' not in self.all_tokens()[tid]:
                self.string_safe_tokens.append(tid)

    def tokens_starting_with(self, prefix: str) -> list[int]:
        """Return token IDs whose stripped text starts with ``prefix``.

        Used by decoder to close one statement.

        Args:
            prefix: The literal prefix to match (e.g. ``","`` or ``"}"``).

        Returns:
            Token IDs whose text, after removing a leading space, begins
            with ``prefix``.
        """
        return [
            token_id
            for token_id in self.all_tokens().keys()
            if self._stripped(token_id).startswith(prefix)
        ]

    @staticmethod
    def _is_number_prefix(s: str) -> bool:
        """Return True if `s` could still become a valid JSON number.

        Accepts: ``""``, ``"-"``, ``"1"``, ``"-1"``, ``"1."``, ``"1.5"``,
                ``"1.5e"``, ``"1.5e-"``, ``"1.5e-3"``
        Rejects: ``"--1"``, ``"1..2"``, ``"abc"``, ``"1.5e-"`` (empty exp)

        Args:
            s: The candidate text.

        Returns:
            True if ``s`` is a valid prefix of a JSON number.
        """
        if s == "":
            return True

        i = 0
        n = len(s)

        if s[i] in "+-":
            i += 1
            if i == n:
                return True

        int_digits = 0
        while i < n and s[i].isdigit():
            i += 1
            int_digits += 1

        if i < n and s[i] == ".":
            i += 1
            if i == n:
                return True
            while i < n and s[i].isdigit():
                i += 1
                int_digits += 1

        if int_digits == 0:
            return False

        if i < n and s[i] in "eE":
            i += 1
            if i == n:
                return True
            if s[i] in "+-":
                i += 1
                if i == n:
                    return True
            exp_digits = 0
            while i < n and s[i].isdigit():
                i += 1
                exp_digits += 1
            if exp_digits == 0:
                return False

        return i == n

    @staticmethod
    def _is_integer_prefix(s: str) -> bool:
        if s == "":
            return True

        i = 0
        n = len(s)

        if s[i] in "+-":
            i += 1
            if i == n:
                return True

        while i < n:
            if not s[i].isdigit():
                return False
            i += 1

        return True

    def tokens_for_type(self, ftype: str, current_text: str) -> list[int]:
        """Return token IDs that can legally continue a value of type `ftype`.

        `current_text` is the text produced so far (e.g. ``"15"`` for a
        number, ``"Hel"`` for a string). The returned list does **not**
        include terminator tokens (``,`` or ``}``) — the decoder adds
        those separately via ``legal_tokens_for_value``.

        Args:
            ftype: One of ``"number"``, ``"string"``, ``"boolean"``.
            current_text: The value text generated so far.

        Returns:
            A list of allowed token IDs.

        Raises:
            ValueError: If ``ftype`` is unknown.
        """
        if ftype == "number":
            return [
                tid
                for tid in self.numeric_tokens
                if self._is_number_prefix(current_text + self._stripped(tid))
            ]

        if ftype == "integer":
            return [
                tid
                for tid in self.numeric_tokens
                if self._is_integer_prefix(current_text + self._stripped(tid))
            ]

        if ftype == "string":
            return list(self.string_safe_tokens)

        if ftype == "boolean":
            seed = (current_text or "t")[0]
            bucket = self.by_first_char.get(seed, [])
            result: list[int] = []
            for tid in bucket:
                candidate = current_text + self._stripped(tid)
                if "true".startswith(candidate) or \
                   "false".startswith(candidate):
                    result.append(tid)
            return result

        raise ValueError(f"Unknown parameter type: {ftype!r}")
