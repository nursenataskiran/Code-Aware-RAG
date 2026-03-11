import ast
from typing import List

from .base_chunker import BaseChunker
from .chunk_models import Chunk
from .utils import (
    read_text_file,
    get_file_name,
    normalize_newlines,
    split_large_text,
)


class ASTChunker(BaseChunker):
    def __init__(self, max_chunk_chars: int = 3500, overlap: int = 300):
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap

    def chunk_file(self, file_path: str, project_name: str) -> List[Chunk]:
        source = normalize_newlines(read_text_file(file_path))
        lines = source.split("\n")

        chunks: List[Chunk] = []
        chunk_index = 0

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._create_file_fallback_chunks(
                source=source,
                lines=lines,
                file_path=str(file_path),
                project_name=project_name,
            )

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                chunk_index = self._add_function_chunk(
                    chunks=chunks,
                    node=node,
                    lines=lines,
                    file_path=str(file_path),
                    project_name=project_name,
                    chunk_index=chunk_index,
                    chunk_type="python_function",
                )

            elif isinstance(node, ast.AsyncFunctionDef):
                chunk_index = self._add_function_chunk(
                    chunks=chunks,
                    node=node,
                    lines=lines,
                    file_path=str(file_path),
                    project_name=project_name,
                    chunk_index=chunk_index,
                    chunk_type="python_async_function",
                )

            elif isinstance(node, ast.ClassDef):
                chunk_index = self._add_class_chunks(
                    chunks=chunks,
                    node=node,
                    lines=lines,
                    file_path=str(file_path),
                    project_name=project_name,
                    chunk_index=chunk_index,
                )

        if not chunks and source.strip():
            return self._create_file_fallback_chunks(
                source=source,
                lines=lines,
                file_path=file_path,
                project_name=project_name,
            )

        return chunks

    def _create_file_fallback_chunks(
        self,
        source: str,
        lines: List[str],
        file_path: str,
        project_name: str,
    ) -> List[Chunk]:
        if not source.strip():
            return []

        parts = split_large_text(
            source.strip(),
            max_chars=self.max_chunk_chars,
            overlap=self.overlap,
        )

        chunks: List[Chunk] = []
        for chunk_index, part in enumerate(parts):
            current_type = "python_file" if chunk_index == 0 else "python_file_part"
            chunks.append(
                Chunk(
                    project_name=project_name,
                    file_path=str(file_path),
                    file_name=get_file_name(file_path),
                    file_type=".py",
                    chunk_type=current_type,
                    chunk_index=chunk_index,
                    text=part,
                    start_line=1,
                    end_line=len(lines),
                )
            )

        return chunks

    def _extract_node_text(self, node: ast.AST, lines: List[str]) -> str:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)

        if start is None or end is None:
            return ""

        return "\n".join(lines[start - 1:end]).strip()

    def _split_if_needed(self, text: str) -> List[str]:
        if len(text) <= self.max_chunk_chars:
            return [text]

        return split_large_text(
            text,
            max_chars=self.max_chunk_chars,
            overlap=self.overlap,
        )

    def _get_method_chunk_type(self, node: ast.AST) -> str:
        decorators = []
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)

        if "staticmethod" in decorators:
            return "python_static_method"
        if "classmethod" in decorators:
            return "python_class_method"
        if isinstance(node, ast.AsyncFunctionDef):
            return "python_async_method"
        return "python_method"

    def _build_class_header_text(self, node: ast.ClassDef, lines: List[str]) -> str:
        method_nodes = [
            child for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if method_nodes:
            header_end = method_nodes[0].lineno - 1
        else:
            header_end = getattr(node, "end_lineno", node.lineno)

        if header_end < node.lineno:
            header_end = node.lineno

        header_text = "\n".join(lines[node.lineno - 1:header_end]).strip()

        if header_text:
            return header_text

        return f"class {node.name}:"

    def _add_function_chunk(
        self,
        chunks: List[Chunk],
        node: ast.AST,
        lines: List[str],
        file_path: str,
        project_name: str,
        chunk_index: int,
        chunk_type: str = "python_function",
        symbol_name: str = None,
        parent_symbol: str = None,
    ) -> int:
        text = self._extract_node_text(node, lines)
        if not text:
            return chunk_index

        parts = self._split_if_needed(text)

        for part_idx, part in enumerate(parts):
            current_type = chunk_type if part_idx == 0 else f"{chunk_type}_part"
            chunks.append(
                Chunk(
                    project_name=project_name,
                    file_path=str(file_path),
                    file_name=get_file_name(file_path),
                    file_type=".py",
                    chunk_type=current_type,
                    chunk_index=chunk_index,
                    text=part,
                    symbol_name=symbol_name or getattr(node, "name", None),
                    parent_symbol=parent_symbol,
                    start_line=getattr(node, "lineno", None),
                    end_line=getattr(node, "end_lineno", None),
                )
            )
            chunk_index += 1

        return chunk_index

    def _add_class_chunks(
        self,
        chunks: List[Chunk],
        node: ast.ClassDef,
        lines: List[str],
        file_path: str,
        project_name: str,
        chunk_index: int,
    ) -> int:
        header_text = self._build_class_header_text(node, lines)

        method_nodes = [
            child for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if header_text:
            first_method_line = (
                method_nodes[0].lineno - 1
                if method_nodes
                else getattr(node, "end_lineno", None)
            )

            chunks.append(
                Chunk(
                    project_name=project_name,
                    file_path=str(file_path),
                    file_name=get_file_name(file_path),
                    file_type=".py",
                    chunk_type="python_class_header",
                    chunk_index=chunk_index,
                    text=header_text,
                    symbol_name=node.name,
                    parent_symbol=None,
                    start_line=getattr(node, "lineno", None),
                    end_line=first_method_line,
                )
            )
            chunk_index += 1

        for method_node in method_nodes:
            method_chunk_type = self._get_method_chunk_type(method_node)
            full_symbol_name = f"{node.name}.{method_node.name}"

            chunk_index = self._add_function_chunk(
                chunks=chunks,
                node=method_node,
                lines=lines,
                file_path=str(file_path),
                project_name=project_name,
                chunk_index=chunk_index,
                chunk_type=method_chunk_type,
                symbol_name=full_symbol_name,
                parent_symbol=node.name,
            )

        return chunk_index