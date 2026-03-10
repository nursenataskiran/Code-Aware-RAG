from abc import ABC, abstractmethod
from typing import List

from .chunk_models import Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk_file(self, file_path: str, project_name: str) -> List[Chunk]:
        """
        Read a file and return a list of Chunk objects.
        """
        raise NotImplementedError