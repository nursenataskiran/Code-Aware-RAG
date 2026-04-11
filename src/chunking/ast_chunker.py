import ast
from typing import List, Optional

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

        # ── Collect module-level preamble (imports, assignments, etc.) ──
        preamble_text = self._build_module_preamble(tree, lines)
        if preamble_text and len(preamble_text.strip()) >= 30:
            chunks.append(
                Chunk(
                    project_name=project_name,
                    file_path=str(file_path),
                    file_name=get_file_name(file_path),
                    file_type=".py",
                    chunk_type="python_module_preamble",
                    chunk_index=chunk_index,
                    text=preamble_text,
                    description=f"Module-level imports and setup in {get_file_name(file_path)} from project {project_name}.",
                )
            )
            chunk_index += 1

        # ── Process top-level definitions ──
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

        # ── Filter out too-short chunks ──
        chunks = [c for c in chunks if not c.is_too_short()]

        return chunks

    # ── Module preamble: imports + top-level assignments ──────────────

    def _build_module_preamble(self, tree: ast.Module, lines: List[str]) -> str:
        """
        Collect all top-level imports, assignments, and standalone
        expressions (not functions/classes) into a module preamble chunk.
        """
        preamble_nodes = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign,
                                 ast.AnnAssign, ast.Expr)):
                # Skip docstrings at module level (Expr with Constant string)
                if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant,)):
                    if isinstance(node.value.value, str):
                        continue
                preamble_nodes.append(node)

        if not preamble_nodes:
            return ""

        preamble_lines = []
        for node in preamble_nodes:
            text = self._extract_node_text(node, lines)
            if text:
                preamble_lines.append(text)

        return "\n".join(preamble_lines)

    # ── Fallback chunks ──────────────────────────────────────────────

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
                    description=f"Python file {get_file_name(file_path)} from project {project_name}.",
                )
            )

        return chunks

    # ── Node text extraction ─────────────────────────────────────────

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

    # ── Semantic description builders ────────────────────────────────

    def _build_function_description(
        self,
        node: ast.AST,
        file_path: str,
        project_name: str,
        chunk_type: str,
        parent_symbol: Optional[str] = None,
    ) -> str:
        """
        Build a natural-language description for a function/method chunk.
        This helps the embedding model bridge the code↔query vocabulary gap.
        """
        name = getattr(node, "name", "unknown")
        file_name = get_file_name(file_path)

        # Extract docstring if present
        docstring = ast.get_docstring(node) or ""
        if docstring:
            # Take first line of docstring
            docstring = docstring.strip().split("\n")[0]

        parts = [f"File: {file_name} | Project: {project_name}"]

        if parent_symbol:
            kind = chunk_type.replace("python_", "").replace("_", " ")
            parts.append(f"Class: {parent_symbol} | {kind}: {name}")
        else:
            kind = chunk_type.replace("python_", "").replace("_", " ")
            parts.append(f"Python {kind}: {name}")

        if docstring:
            parts.append(f"Description: {docstring}")

        return " | ".join(parts)

    # ── Method type detection ────────────────────────────────────────

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

    # ── Class overview builder ───────────────────────────────────────

    def _build_class_overview_text(self, node: ast.ClassDef, lines: List[str]) -> str:
        """
        Build a rich class overview that includes:
        - The class definition line
        - The class docstring (if any)
        - Class-level attributes
        - A list of all method signatures
        
        This replaces the near-empty 'class Foo:' header chunks.
        """
        parts = []

        # Class definition line
        class_line = lines[node.lineno - 1].strip()
        parts.append(class_line)

        # Class docstring
        docstring = ast.get_docstring(node)
        if docstring:
            parts.append(f'    """{docstring}"""')

        # Class-level attributes (Assign nodes in class body)
        for child in node.body:
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                attr_text = self._extract_node_text(child, lines)
                if attr_text:
                    parts.append(f"    {attr_text}")

        # Method signatures (just the def line, not the body)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_line = lines[child.lineno - 1].strip()
                method_docstring = ast.get_docstring(child)
                if method_docstring:
                    first_doc_line = method_docstring.strip().split("\n")[0]
                    parts.append(f"    {method_line}  # {first_doc_line}")
                else:
                    parts.append(f"    {method_line}")

        return "\n".join(parts)

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

    # ── Function chunk builder ───────────────────────────────────────

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

        # Build semantic description
        description = self._build_function_description(
            node=node,
            file_path=file_path,
            project_name=project_name,
            chunk_type=chunk_type,
            parent_symbol=parent_symbol,
        )

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
                    description=description,
                )
            )
            chunk_index += 1

        return chunk_index

    # ── Class chunk builder ──────────────────────────────────────────

    def _add_class_chunks(
        self,
        chunks: List[Chunk],
        node: ast.ClassDef,
        lines: List[str],
        file_path: str,
        project_name: str,
        chunk_index: int,
    ) -> int:
        # Build a rich class overview instead of the near-empty header
        overview_text = self._build_class_overview_text(node, lines)

        method_nodes = [
            child for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if overview_text and len(overview_text.strip()) >= 30:
            class_docstring = ast.get_docstring(node) or ""
            first_doc_line = class_docstring.strip().split("\n")[0] if class_docstring else ""

            description = f"File: {get_file_name(file_path)} | Project: {project_name} | Class: {node.name}"
            if first_doc_line:
                description += f" | {first_doc_line}"
            if method_nodes:
                method_names = [m.name for m in method_nodes]
                description += f" | Methods: {', '.join(method_names)}"

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
                    chunk_type="python_class_overview",
                    chunk_index=chunk_index,
                    text=overview_text,
                    symbol_name=node.name,
                    parent_symbol=None,
                    start_line=getattr(node, "lineno", None),
                    end_line=first_method_line,
                    description=description,
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