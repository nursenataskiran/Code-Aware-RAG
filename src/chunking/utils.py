from pathlib import Path
from typing import List


def read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    return Path(file_path).read_text(encoding=encoding)


def get_file_name(file_path: str) -> str:
    return Path(file_path).name


def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix.lower()


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_large_text(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200
) -> List[str]:
    """
    Splits very large text into smaller overlapping pieces.
    Character-based for simplicity.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]
    
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end == text_length:
            break

        next_start = end - overlap
        if next_start <= start:
            break

        start = next_start

    return pieces