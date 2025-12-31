"""Text chunking utilities for splitting large texts."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    """Calculate the chunk size based on the token count and token limit.

    Args:
        token_count: The total number of tokens.
        token_limit: The maximum number of tokens allowed per chunk.

    Returns:
        The calculated chunk size.

    Description:
        This function calculates the chunk size based on the given token count
        and token limit. If the token count is less than or equal to the token
        limit, the function returns the token count as the chunk size.
        Otherwise, it calculates the number of chunks needed to accommodate
        all the tokens within the token limit.

    Example:
        >>> calculate_chunk_size(1000, 500)
        500
        >>> calculate_chunk_size(1530, 500)
        389
        >>> calculate_chunk_size(2242, 500)
        496
    """
    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    model_name: str = "gpt-4",
) -> list[str]:
    """Split text into chunks using tiktoken encoder.

    Args:
        text: The text to split.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        model_name: Model name for tiktoken encoder.

    Returns:
        List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(text)
