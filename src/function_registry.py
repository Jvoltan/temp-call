"""Registry mapping function names to their validated definitions.

Wraps the raw list of function dictionaries loaded from
``functions_definition.json`` and exposes helpers used by the decoder
to look up names, descriptions, parameters, and return types.
"""
try:
    from typing import Any
    import sys
    from box import Box
except ModuleNotFoundError:
    print("BRUHHHHHHHHH, I saw you brada !!!")
    sys.exit(130)


class FunctionRegistration:
    """Read-only view over the list of function definitions.

    Attributes:
        functions_list: Raw list of function dictionaries as loaded
            from the JSON file.
    """
    def __init__(self, functions_list: list[dict[str, Any]]) -> None:
        """Store the raw function list.

        Args:
            functions_list: Raw list of function dictionaries.
        """
        self.functions_list = functions_list

    def get_all_functions(self) -> list[str]:
        """Return the names of all registered functions.

        Returns:
            A list of function-name strings in declaration order.
        """
        self.list_function_name: list[str] = []
        for dict_content in self.functions_list:
            for key, value in dict_content.items():
                if key == "name":
                    self.list_function_name.append(value)
        return self.list_function_name

    def get_function(self, func_name: str) -> Box:
        """Return the definition of a function as an attribute-access Box.

        Args:
            func_name: The function name to look up.

        Returns:
            A ``Box`` wrapping the matching function dictionary so its
            fields can be accessed with dot notation.

        Raises:
            ValueError: If no function named ``func_name`` is registered.
        """
        for function in self.functions_list:
            for value in function.values():
                if func_name == value:
                    real_value = Box(function)
                    return real_value
        raise ValueError(f"ERROR - {func_name} function not found")
