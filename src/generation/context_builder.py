from typing import List

from src.retrieval.retriever import RetrievalResult


def format_source(result: RetrievalResult, rank: int) -> str:
    meta = result.metadata

    project = meta.get("project_name", "Unknown")
    file_name = meta.get("file_name", "Unknown")
    chunk_type = meta.get("chunk_type", "Unknown")
    symbol = meta.get("symbol_name", "None")
    start_line = meta.get("start_line")
    end_line = meta.get("end_line")

    lines = ""
    if start_line is not None and end_line is not None:
        lines = f"Lines: {start_line}-{end_line}\n"

    source_block = (
        f"[Source {rank}]\n"
        f"Project: {project}\n"
        f"File: {file_name}\n"
        f"Chunk Type: {chunk_type}\n"
        f"Symbol: {symbol}\n"
        f"{lines}\n"
        f"{result.text.strip()}\n"
    )

    return source_block


def build_context(results: List[RetrievalResult]) -> str:
    if not results:
        return "No relevant context found."

    formatted_sources = [
        format_source(result, rank=i)
        for i, result in enumerate(results, start=1)
    ]

    return "\n\n".join(formatted_sources)