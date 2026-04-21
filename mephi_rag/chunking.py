# подготовка текста, чанки для индекса и bm25

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_markdown_by_separator_for_rag(
    file_path: str,
    separator: str = "—————",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
):
    # исходный файл целиком
    text = Path(file_path).read_text(encoding="utf-8")

    # крупные сегменты по разделителю
    raw_blocks = [block.strip() for block in text.split(separator)]
    raw_blocks = [block for block in raw_blocks if block]

    # один документ langchain на каждый сегмент
    block_docs = [
        Document(
            page_content=block,
            metadata={"block_id": i + 1},
        )
        for i, block in enumerate(raw_blocks)
    ]

    # нарезка сегментов на чанки фиксированной длины
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    final_chunks = []
    for doc in block_docs:
        chunks = splitter.create_documents(
            [doc.page_content],
            metadatas=[doc.metadata],
        )

        # метка позиции чанка внутри сегмента
        for j, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_in_block"] = j
            chunk.metadata["total_chunks_in_block"] = len(chunks)

        final_chunks.extend(chunks)

    return final_chunks
