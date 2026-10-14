"""Terminal-only decoration helpers.

Nothing in this module ever writes to the JSON output file. All
functions print to stdout or stderr and are safe to call at any
point in the pipeline.
"""
import sys


def _supports_color() -> bool:
    """Return True if stdout is a TTY and colors should be used.

    Returns:
        True when the current stdout supports ANSI escape codes.
    """
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _C:
    """ANSI color codes, empty strings when colors are disabled."""

    RESET = "\033[0m" if _supports_color() else ""
    BOLD = "\033[1m" if _supports_color() else ""
    DIM = "\033[2m" if _supports_color() else ""
    CYAN = "\033[36m" if _supports_color() else ""
    GREEN = "\033[32m" if _supports_color() else ""
    YELLOW = "\033[33m" if _supports_color() else ""
    RED = "\033[31m" if _supports_color() else ""


BANNER = r"""
  o-o     o o      o   o          o   o          o
 /        | |      |\ /|          |\ /|          |
O      oo | |      | O | o-o      | O |  oo o  o O-o  o-o
 \    | | | |      |   | |-'      |   | | | |  | |  | |-'
  o-o o-o-o o      o   o o-o      o   o o-o-o--O o-o  o-o
              o---o          o---o             |
                                            o--o
"""


def print_banner(version: str = "0.1.0") -> None:
    """Print the program banner once at startup.

    Args:
        version: Version string shown under the banner.
    """
    print(f"{_C.CYAN}{BANNER}{_C.RESET}", end="")
    print(f"{_C.DIM}  call-me-maybe  v{version}{_C.RESET}")
    print()


def print_section(title: str) -> None:
    """Print a section header line.

    Args:
        title: The section title to display.
    """
    width = 60
    bar = "─" * width
    print(f"\n{_C.BOLD}{bar}{_C.RESET}")
    print(f"{_C.BOLD}  {title}{_C.RESET}")
    print(f"{_C.BOLD}{bar}{_C.RESET}")


def print_prompt(index: int, total: int, prompt: str) -> None:
    """Print a per-prompt progress line.

    Args:
        index: 1-based index of the current prompt.
        total: Total number of prompts.
        prompt: The natural-language prompt being processed.
    """
    counter = f"[{index:>{len(str(total))}}/{total}]"
    print(f"{_C.CYAN}{counter}{_C.RESET} {prompt}")


def print_result(name: str, parameters: dict[str, object]) -> None:
    """Print the resolved function call for one prompt.

    Args:
        name: The function name chosen by the model.
        parameters: The argument dict produced by the decoder.
    """
    params = ", ".join(f"{k}={v!r}" for k, v in parameters.items())
    print(f"        {_C.GREEN}→{_C.RESET} {_C.BOLD}{name}{_C.RESET}({params})")


def print_error(message: str) -> None:
    """Print an error line to stderr.

    Args:
        message: The error text.
    """
    print(f"{_C.RED}error:{_C.RESET} {message}", file=sys.stderr)


def print_summary(processed: int, output_path: str) -> None:
    """Print the final summary line.

    Args:
        processed: Number of prompts processed.
        output_path: Path to the JSON file that was written.
    """
    print()
    print(f"{_C.GREEN}✓{_C.RESET} "
          f"Processed {_C.BOLD}{processed}{_C.RESET} prompt(s).")
    print(f"{_C.DIM}  Output written to {output_path}{_C.RESET}")
