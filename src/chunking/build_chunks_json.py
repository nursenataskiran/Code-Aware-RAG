from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from src.chunking.smart_chunker import SmartChunker
from src.config import RAW_DATA_DIR, CHUNKS_PATH

OUTPUT_FILE = CHUNKS_PATH


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

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    for project_dir in sorted(RAW_DATA_DIR.iterdir()):
        if not project_dir.is_dir():
            continue

        print(f"\nProcessing project: {project_dir.name}")

        files = sorted(collect_supported_files(project_dir))

        for file_path in files:
            try:
                chunks = chunker.chunk_file(file_path, project_dir.name)

                for chunk in chunks:
                    all_chunks.append(chunk.to_dict())

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
