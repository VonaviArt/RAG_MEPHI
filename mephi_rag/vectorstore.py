# хранилище chroma на диске

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def _chroma_has_data(persist_directory: Path) -> bool:
    # каталог не пустой — коллекция уже есть
    return persist_directory.exists() and any(persist_directory.iterdir())


def make_vector_db(
    chunk_documents: list[Document],
    hf_embeddings_model: HuggingFaceEmbeddings,
    persist_directory: str,
) -> Chroma:
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    # открыть существующий индекс
    if _chroma_has_data(persist_directory):
        return Chroma(
            persist_directory=str(persist_directory),
            embedding_function=hf_embeddings_model,
        )

    # создать индекс из чанков
    return Chroma.from_documents(
        documents=chunk_documents,
        embedding=hf_embeddings_model,
        persist_directory=str(persist_directory),
    )
