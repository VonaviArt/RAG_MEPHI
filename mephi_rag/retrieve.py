# гибридный поиск

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

_ENSEMBLE_WEIGHTS = [0.3, 0.7]


def make_ensemble_retriever(
    vector_db: Chroma,
    chunk_documents: list[Document],
    k: int = 20,
):
    # векторный top-k
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": k})

    # лексический bm25 по тем же чанкам
    bm25_retriever = BM25Retriever.from_documents(chunk_documents)
    bm25_retriever.k = k

    # слияние двух списков кандидатов
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=_ENSEMBLE_WEIGHTS,
    )
