"""Token counting utilities for text processing."""

import tiktoken


MAX_TOKENS_PER_CHUNK = 1000


def num_tokens_in_string(
    input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    """Calculate the number of tokens in a given string using a specified encoding.

    Args:
        input_str: The input string to be tokenized.
        encoding_name: The name of the encoding to use. Defaults to "cl100k_base",
            which is the most commonly used encoder (used by GPT-4).

    Returns:
        The number of tokens in the input string.

    Example:
        >>> text = "Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)
        >>> print(num_tokens)
        5
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens
