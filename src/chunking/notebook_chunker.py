import json
from typing import List

from .base_chunker import BaseChunker
from .chunk_models import Chunk
from .utils import (
    get_file_name,
    split_large_text,
)


class NotebookChunker(BaseChunker):
    def __init__(
        self,
        max_chunk_chars: int = 2500,
        overlap: int = 200,
        min_code_cell_chars: int = 160,
        merge_target_chars: int = 1500,
        markdown_context_chars: int = 300,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap
        self.min_code_cell_chars = min_code_cell_chars
        self.merge_target_chars = merge_target_chars
        self.markdown_context_chars = markdown_context_chars

    def chunk_file(self, file_path: str, project_name: str) -> List[Chunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])
        chunks: List[Chunk] = []
        chunk_index = 0
        current_section = None
        last_markdown_context = None

        cell_index = 0
        while cell_index < len(cells):
            cell = cells[cell_index]
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", [])).strip()

            if not source:
                cell_index += 1
                continue

            if cell_type == "markdown":
                header = self._extract_first_markdown_header(source)
                if header:
                    current_section = header
                last_markdown_context = self._summarize_markdown_context(source)

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

                    # Build semantic description for markdown cells
                    description = f"Notebook: {get_file_name(file_path)} | Project: {project_name}"
                    if current_section:
                        description += f" | Section: {current_section}"

                    chunks.append(
                        Chunk(
                            project_name=project_name,
                            file_path=str(file_path),
                            file_name=get_file_name(file_path),
                            file_type=".ipynb",
                            chunk_type=current_type,
                            chunk_index=chunk_index,
                            text=part,
                            cell_index=cell_index,
                            cell_type="markdown",
                            section_header=current_section,
                            description=description,
                        )
                    )
                    chunk_index += 1
                cell_index += 1

            elif cell_type == "code":
                code_source, consumed_until = self._merge_short_code_cells(
                    cells=cells,
                    start_index=cell_index,
                    initial_source=source,
                )
                # Always attach section context — removed the length guard
                code_text = self._attach_code_context(
                    code_source=code_source,
                    current_section=current_section,
                    markdown_context=last_markdown_context,
                )
                parts = split_large_text(
                    code_text,
                    max_chars=self.max_chunk_chars,
                    overlap=self.overlap,
                )

                for part_idx, part in enumerate(parts):
                    current_type = (
                        "notebook_code"
                        if part_idx == 0
                        else "notebook_code_part"
                    )

                    # Build semantic description for code cells
                    description = f"Notebook code: {get_file_name(file_path)} | Project: {project_name}"
                    if current_section:
                        description += f" | Section: {current_section}"
                    if last_markdown_context:
                        description += f" | Context: {last_markdown_context[:100]}"

                    chunks.append(
                        Chunk(
                            project_name=project_name,
                            file_path=str(file_path),
                            file_name=get_file_name(file_path),
                            file_type=".ipynb",
                            chunk_type=current_type,
                            chunk_index=chunk_index,
                            text=part,
                            cell_index=cell_index,
                            cell_type="code",
                            section_header=current_section,
                            description=description,
                        )
                    )
                    chunk_index += 1
                cell_index = consumed_until + 1
            else:
                cell_index += 1

        # Filter out too-short chunks
        chunks = [c for c in chunks if not c.is_too_short()]

        return chunks

    def _extract_first_markdown_header(self, text: str):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:80]
        return None

    def _summarize_markdown_context(self, text: str) -> str | None:
        lines = []
        for line in text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                lines.append(stripped)
            if sum(len(item) for item in lines) >= self.markdown_context_chars:
                break

        if not lines:
            return None

        summary = " ".join(lines)
        return summary[:self.markdown_context_chars]

    def _merge_short_code_cells(
        self,
        cells: List[dict],
        start_index: int,
        initial_source: str,
    ) -> tuple[str, int]:
        if len(initial_source) >= self.min_code_cell_chars:
            return initial_source, start_index

        merged_parts = [initial_source]
        current_index = start_index

        while current_index + 1 < len(cells):
            next_cell = cells[current_index + 1]
            if next_cell.get("cell_type") != "code":
                break

            next_source = "".join(next_cell.get("source", [])).strip()
            if not next_source:
                current_index += 1
                continue

            projected_length = len("\n\n".join(merged_parts + [next_source]))
            if projected_length > self.merge_target_chars:
                break

            merged_parts.append(next_source)
            current_index += 1

            if projected_length >= self.min_code_cell_chars:
                break

        return "\n\n".join(merged_parts), current_index

    def _attach_code_context(
        self,
        code_source: str,
        current_section: str | None,
        markdown_context: str | None,
    ) -> str:
        """
        Always attach section context to code cells.
        Previously this was gated on code length — now it always applies.
        """
        prefix_lines = []

        if current_section:
            prefix_lines.append(f"# Section: {current_section}")

        if markdown_context:
            prefix_lines.append(f"# Context: {markdown_context}")

        if not prefix_lines:
            return code_source

        return "\n".join(prefix_lines) + "\n\n" + code_source
