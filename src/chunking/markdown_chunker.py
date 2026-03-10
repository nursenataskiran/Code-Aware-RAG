import re
from typing import List, Tuple

from .base_chunker import BaseChunker
from .chunk_models import Chunk
from .utils import (
    read_text_file,
    get_file_name,
    normalize_newlines,
    split_large_text,
)


class MarkdownChunker(BaseChunker):
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def __init__(self, max_chunk_chars: int = 3500, overlap: int = 350):
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap

    def chunk_file(self, file_path: str, project_name: str) -> List[Chunk]:
        text = normalize_newlines(read_text_file(file_path))
        sections = self._split_by_headings(text)

        chunks: List[Chunk] = []
        chunk_index = 0

        if not sections:
            parts = split_large_text(
                text,
                max_chars=self.max_chunk_chars,
                overlap=self.overlap,
            )
            for part in parts:
                chunks.append(
                    Chunk(
                        project_name=project_name,
                        file_path=file_path,
                        file_name=get_file_name(file_path),
                        file_type=".md",
                        chunk_type="markdown_document",
                        chunk_index=chunk_index,
                        text=part,
                    )
                )
                chunk_index += 1
            return chunks

        for header, content in sections:
            section_text = content.strip()
            if not section_text:
                continue

            parts = split_large_text(
                section_text,
                max_chars=self.max_chunk_chars,
                overlap=self.overlap,
            )

            for part_idx, part in enumerate(parts):
                current_type = "markdown_section" if part_idx == 0 else "markdown_section_part"
                chunks.append(
                    Chunk(
                        project_name=project_name,
                        file_path=file_path,
                        file_name=get_file_name(file_path),
                        file_type=".md",
                        chunk_type=current_type,
                        chunk_index=chunk_index,
                        text=part,
                        section_header=header,
                    )
                )
                chunk_index += 1

        return chunks

    def _split_by_headings(self, text: str) -> List[Tuple[str, str]]:
        matches = list(self.HEADING_PATTERN.finditer(text))

        if not matches:
            return []

        sections: List[Tuple[str, str]] = []

        first_heading_start = matches[0].start()
        intro_text = text[:first_heading_start].strip()
        if intro_text:
            sections.append(("Introduction", intro_text))

        for i, match in enumerate(matches):
            header_text = match.group(2).strip()
            content_start = match.start()

            if i + 1 < len(matches):
                content_end = matches[i + 1].start()
            else:
                content_end = len(text)

            section_content = text[content_start:content_end].strip()
            sections.append((header_text, section_content))

        return sections