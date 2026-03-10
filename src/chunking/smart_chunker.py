from typing import List

from .base_chunker import BaseChunker
from .chunk_models import Chunk
from .ast_chunker import ASTChunker
from .markdown_chunker import MarkdownChunker
from .notebook_chunker import NotebookChunker
from .utils import get_file_extension


class SmartChunker(BaseChunker):
    def __init__(self):
        self.ast_chunker = ASTChunker()
        self.markdown_chunker = MarkdownChunker()
        self.notebook_chunker = NotebookChunker()

    def chunk_file(self, file_path: str, project_name: str) -> List[Chunk]:
        extension = get_file_extension(file_path)

        if extension == ".py":
            return self.ast_chunker.chunk_file(file_path, project_name)

        if extension == ".md":
            return self.markdown_chunker.chunk_file(file_path, project_name)

        if extension == ".ipynb":
            return self.notebook_chunker.chunk_file(file_path, project_name)

        raise ValueError(f"Unsupported file type for chunking: {extension}")