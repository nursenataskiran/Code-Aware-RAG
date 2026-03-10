from dataclasses import dataclass, asdict
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)