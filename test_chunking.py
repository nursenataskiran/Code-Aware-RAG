from pathlib import Path
from collections import Counter

from src.chunking.smart_chunker import SmartChunker


FILES = [
    "data/raw/Formula1_Race_Prediction/data_cleaning (1).py",
    "data\Raw\Human_Stress_Prediction\data_preprocessor.py"
]

SHOW_FULL_TEXT = True          # False yaparsan sadece preview gösterir
PREVIEW_CHARS = 220            # Full text kapalıysa gösterilecek önizleme uzunluğu
ONLY_PYTHON = True             # Sadece .py dosyalarını test et


def shorten_text(text: str, limit: int = 220) -> str:
    text = text.strip()
    text = text.replace("\t", "    ")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def format_range(start_line, end_line) -> str:
    if start_line is None and end_line is None:
        return "-"
    if start_line == end_line:
        return str(start_line)
    return f"{start_line}-{end_line}"


def print_chunk(chunk, idx: int, total: int) -> None:
    print("\n" + "-" * 90)
    print(f"CHUNK {idx}/{total}")
    print("-" * 90)

    print(f"type         : {chunk.chunk_type}")
    print(f"symbol       : {chunk.symbol_name or '-'}")
    print(f"parent       : {chunk.parent_symbol or '-'}")
    print(f"lines        : {format_range(chunk.start_line, chunk.end_line)}")
    print(f"length(chars): {len(chunk.text)}")
    print(f"file         : {chunk.file_name}")
    print(f"section      : {chunk.section_header or '-'}")
    print(f"cell         : {chunk.cell_type or '-'} / {chunk.cell_index if chunk.cell_index is not None else '-'}")

    if SHOW_FULL_TEXT:
        print("\nTEXT:\n")
        print(chunk.text)
    else:
        print("\nPREVIEW:\n")
        print(shorten_text(chunk.text, PREVIEW_CHARS))


def print_summary(chunks) -> None:
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    type_counts = Counter(chunk.chunk_type for chunk in chunks)

    print(f"total chunks : {len(chunks)}")
    print("chunk types  :")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  - {chunk_type}: {count}")

    symbols = [chunk.symbol_name for chunk in chunks if chunk.symbol_name]
    if symbols:
        print("\nsymbols:")
        for sym in symbols:
            print(f"  - {sym}")


def main() -> None:
    chunker = SmartChunker()

    files = FILES
    if ONLY_PYTHON:
        files = [f for f in FILES if Path(f).suffix == ".py"]

    if not files:
        print("Test edilecek .py dosyası bulunamadı.")
        return

    for file in files:
        print("\n" + "=" * 90)
        print(f"FILE: {file}")
        print("=" * 90)

        project_name = Path(file).parent.name
        chunks = chunker.chunk_file(file, project_name)

        if not chunks:
            print("Bu dosya için chunk üretilmedi.")
            continue

        for i, chunk in enumerate(chunks, start=1):
            print_chunk(chunk, i, len(chunks))

        print_summary(chunks)


if __name__ == "__main__":
    main()