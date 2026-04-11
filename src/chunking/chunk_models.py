from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class Chunk:
    project_name: str
    file_path: str
    file_name: str
    file_type: str
    chunk_type: str
    chunk_index: int
    text: str

    symbol_name: Optional[str] = None
    parent_symbol: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    cell_index: Optional[int] = None
    cell_type: Optional[str] = None
    section_header: Optional[str] = None

    # Semantic description prepended to text for embedding (but stored separately)
    description: Optional[str] = None

    # Minimum chunk size — chunks shorter than this are filtered out
    MIN_CHUNK_CHARS: int = field(default=30, repr=False, compare=False)

    def build_id(self) -> str:
        symbol = self._sanitize_id_part(self.symbol_name or "no_symbol")
        file_part = self._sanitize_id_part(Path(self.file_path).with_suffix("").as_posix())
        project_part = self._sanitize_id_part(self.project_name)
        chunk_type = self._sanitize_id_part(self.chunk_type)
        return f"{project_part}__{file_part}__{chunk_type}__{self.chunk_index}__{symbol}"

    @staticmethod
    def _sanitize_id_part(value: str) -> str:
        cleaned = value.replace("\\", "/").replace(" ", "_")
        return "".join(char if char.isalnum() or char in {"_", "-", "/"} else "_" for char in cleaned)

    def is_too_short(self) -> bool:
        """Check if this chunk is too short to be useful for retrieval."""
        return len(self.text.strip()) < self.MIN_CHUNK_CHARS

    def embedding_text(self) -> str:
        """
        Text used for embedding. Prepends the semantic description
        to the raw text so the embedding captures both the natural-language
        summary and the actual code/content.
        """
        if self.description:
            return f"{self.description}\n\n{self.text}"
        return self.text

    def metadata(self) -> Dict[str, Any]:
        meta = {
            "project_name": self.project_name,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "symbol_name": self.symbol_name,
            "parent_symbol": self.parent_symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "cell_index": self.cell_index,
            "cell_type": self.cell_type,
            "section_header": self.section_header,
            "description": self.description,
        }
        return {k: v for k, v in meta.items() if v is not None}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.build_id(),
            "text": self.text,
            "embedding_text": self.embedding_text(),
            "metadata": self.metadata()
        }
