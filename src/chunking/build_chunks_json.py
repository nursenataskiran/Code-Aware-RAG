from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from src.chunking.smart_chunker import SmartChunker
from src.chunking.chunk_models import Chunk

RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/chunks.json")


def generate_chunk_id(chunk: Chunk) -> str:
    project = chunk.project_name.replace(" ", "_")
    file_stem = Path(chunk.file_name).stem.replace(" ", "_")
    chunk_type = chunk.chunk_type.replace(" ", "_")

    symbol_part = chunk.symbol_name if chunk.symbol_name else "no_symbol"
    symbol_part = symbol_part.replace(" ", "_")

    return f"{project}__{file_stem}__{chunk_type}__{chunk.chunk_index}__{symbol_part}"


def collect_supported_files(project_dir: Path) -> List[Path]:
    supported_extensions = {".py", ".md", ".ipynb"}
    files: List[Path] = []

    for path in project_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_extensions:
            files.append(path)

    return files


def build_chunks_json() -> List[Dict[str, Any]]:
    chunker = SmartChunker()
    all_chunks: List[Dict[str, Any]] = []

    for project_dir in RAW_DATA_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        print(f"\nProcessing project: {project_dir.name}")

        files = collect_supported_files(project_dir)

        for file_path in files:
            try:
                chunks = chunker.chunk_file(file_path, project_dir.name)

                for chunk in chunks:
                    chunk_dict = chunk.to_dict()
                    chunk_dict["id"] = generate_chunk_id(chunk)
                    all_chunks.append(chunk_dict)

                print(f"  OK -> {file_path} | chunks: {len(chunks)}")

            except Exception as e:
                print(f"  FAILED -> {file_path} | error: {e}")

    return all_chunks


def save_chunks(chunks: List[Dict[str, Any]]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(chunks)} chunks to: {OUTPUT_FILE}")


def main() -> None:
    chunks = build_chunks_json()
    save_chunks(chunks)


if __name__ == "__main__":
    main()