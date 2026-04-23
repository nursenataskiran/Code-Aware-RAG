from src.chunking.smart_chunker import SmartChunker

files = [
    "data/raw/Formula1_Race_Prediction/data_cleaning (1).py",
    "data/raw/Formula1_Race_Prediction/README.md",
    "data/raw/Formula1_Race_Prediction/lstm.ipynb",
]

chunker = SmartChunker()

for file in files:

    print("\n" + "="*80)
    print("FILE:", file)
    print("="*80)

    project_name = file.split("/")[-2]

    chunks = chunker.chunk_file(file, project_name)

    for i, chunk in enumerate(chunks):

        print("\n-------------------------------")
        print(f"CHUNK {i+1}/{len(chunks)}")
        print("type:", chunk.chunk_type)
        print("symbol:", chunk.symbol_name)
        print("section:", chunk.section_header)
        print("length:", len(chunk.text))
        print("\nTEXT:\n")
        print(chunk.text)