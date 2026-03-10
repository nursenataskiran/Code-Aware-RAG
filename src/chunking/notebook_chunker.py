import json
from typing import List

from .base_chunker import BaseChunker
from .chunk_models import Chunk
from .utils import (
    get_file_name,
    split_large_text,
)


class NotebookChunker(BaseChunker):
    def __init__(self, max_chunk_chars: int = 2500, overlap: int = 200):
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap

    def chunk_file(self, file_path: str, project_name: str) -> List[Chunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])
        chunks: List[Chunk] = []
        chunk_index = 0
        current_section = None

        for cell_index, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", [])).strip()

            if not source:
                continue

            if cell_type == "markdown":
                header = self._extract_first_markdown_header(source)
                if header:
                    current_section = header

                parts = split_large_text(
                    source,
                    max_chars=self.max_chunk_chars,
                    overlap=self.overlap,
                )

                for part_idx, part in enumerate(parts):
                    current_type = (
                        "notebook_markdown"
                        if part_idx == 0
                        else "notebook_markdown_part"
                    )

                    chunks.append(
                        Chunk(
                            project_name=project_name,
                            file_path=file_path,
                            file_name=get_file_name(file_path),
                            file_type=".ipynb",
                            chunk_type=current_type,
                            chunk_index=chunk_index,
                            text=part,
                            cell_index=cell_index,
                            cell_type="markdown",
                            section_header=current_section,
                        )
                    )
                    chunk_index += 1

            elif cell_type == "code":
                parts = split_large_text(
                    source,
                    max_chars=self.max_chunk_chars,
                    overlap=self.overlap,
                )

                for part_idx, part in enumerate(parts):
                    current_type = (
                        "notebook_code"
                        if part_idx == 0
                        else "notebook_code_part"
                    )

                    chunks.append(
                        Chunk(
                            project_name=project_name,
                            file_path=file_path,
                            file_name=get_file_name(file_path),
                            file_type=".ipynb",
                            chunk_type=current_type,
                            chunk_index=chunk_index,
                            text=part,
                            cell_index=cell_index,
                            cell_type="code",
                            section_header=current_section,
                        )
                    )
                    chunk_index += 1

        return chunks

    def _extract_first_markdown_header(self, text: str):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return None