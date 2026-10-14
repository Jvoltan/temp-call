"""Pydantic models for the function-calling pipeline.

Defines the validated containers used to parse the input JSON files
and to serialise the final function-call output.
"""
try:
    import sys
    from typing import Any
    from pydantic import BaseModel, Field, ConfigDict
    from enum import Enum
except ModuleNotFoundError:
    print("BRUHHHHHHHHH, I saw you brada !!!")
    sys.exit(130)


class _Value(Enum):
    """Enumeration of the JSON-schema scalar types supported by the tool."""
    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    BOOL = "boolean"


class _FunctionParameter(BaseModel):
    """Schema for a single parameter entry in a function definition."""
    param_types: _Value = Field(alias="type")
    model_config = ConfigDict(extra="forbid")


class FunctionDefinition(BaseModel):
    """Validated representation of one entry in ``functions_definition.json``.

    Attributes:
        name: Function identifier (e.g. ``"fn_add_numbers"``).
        description: Human-readable description of the function.
        parameters: Mapping ``param_name -> _FunctionParameter``.
        returns: Mapping with a single ``"type"`` key describing the return.

    Raises:
        pydantic.ValidationError: If required keys are missing, unknown
            keys are present, or a type value is not one of
            ``"number"``/``"string"``/``"boolean"``.
    """
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, _FunctionParameter] = Field(..., min_length=1)
    returns: dict[str, _Value] = Field(..., min_length=1, max_length=1)
    model_config = ConfigDict(extra="forbid")


class FunctionCallTest(BaseModel):
    """A single natural-language prompt from ``function_calling_tests.json``.

    Attributes:
        prompt: The user's natural-language request.
    """
    prompt: str = Field(..., min_length=1)
    model_config = ConfigDict(extra="forbid")


class FunctionCall(BaseModel):
    """Final structured output: one resolved function call.

    Attributes:
        prompt: The original natural-language request.
        name: The name of the function selected by the model.
        parameters: Argument values keyed by parameter name.
    """
    prompt: str
    name: str
    parameters: dict[str, Any]
