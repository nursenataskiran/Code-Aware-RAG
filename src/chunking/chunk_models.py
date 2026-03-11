from dataclasses import dataclass
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

    def build_id(self) -> str:
        symbol = self.symbol_name or "nosymbol"
        file_clean = self.file_name.replace(".", "_")
        symbol_clean = symbol.replace(".", "_")
        return f"{self.project_name}__{file_clean}__{self.chunk_type}__{self.chunk_index}__{symbol_clean}"

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
        }
        return {k: v for k, v in meta.items() if v is not None}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.build_id(),
            "text": self.text,
            "metadata": self.metadata()
        }