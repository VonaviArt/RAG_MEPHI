# оркестрация rag: env - чанки - chroma - поиск - реранк - llm

import os
import shutil
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from mephi_rag.chunking import split_markdown_by_separator_for_rag
from mephi_rag.embeddings import hf_embeddings_model
from mephi_rag.generate import chat_answer
from mephi_rag.rerank import ChunkInfo, build_context_after_rerank, make_reranker
from mephi_rag.retrieve import make_ensemble_retriever
from mephi_rag.vectorstore import make_vector_db

# корень репозитория для относительных путей
REPO_ROOT = Path(__file__).resolve().parent.parent

# кэш в памяти, ленивая инициализация
_inited = False
_ensemble_retriever = None
_reranker = None


class QueryResult(TypedDict):
    answer: str
    chunks: list[ChunkInfo]


def _resolve_under_repo(name: str, default: str) -> Path:
    # путь из env; относительные — от корня репо
    raw = os.getenv(name) or default
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _init(force_rebuild: bool = False):
    global _inited, _ensemble_retriever, _reranker

    # загрузка .env
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()

    HF_API_KEY = os.getenv("HF_API_KEY")
    if HF_API_KEY:
        os.environ["HF_TOKEN"] = HF_API_KEY

    rag_md = _resolve_under_repo("RAG_MD_PATH", "raw_data/RAG.md")
    if not rag_md.is_file():
        raise FileNotFoundError("Не найден RAG.md: %s" % rag_md)

    persist_directory = _resolve_under_repo("CHROMA_PERSIST_DIR", "chroma_db")
    # при пересборке — удалить старый каталог chroma
    if force_rebuild and persist_directory.exists():
        shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

    # корпус чанков из RAG.md
    chunk_documents = split_markdown_by_separator_for_rag(
        file_path=str(rag_md),
        separator="—————",
        chunk_size=1000,
        chunk_overlap=100,
    )

    emb = hf_embeddings_model()
    vector_db = make_vector_db(
        chunk_documents,
        emb,
        str(persist_directory),
    )

    _ensemble_retriever = make_ensemble_retriever(vector_db, chunk_documents, k=20)
    # reranker создаётся при первом get_answer (тяжёлая модель)
    _reranker = None
    _inited = True


def reset_pipeline():
    global _inited, _ensemble_retriever, _reranker
    # только память; каталог chroma на диске не трогаем
    _inited = False
    _ensemble_retriever = None
    _reranker = None


def rebuild_index():
    # новый индекс на диске и заново поднять пайплайн
    reset_pipeline()
    _init(force_rebuild=True)
    return _resolve_under_repo("CHROMA_PERSIST_DIR", "chroma_db")


def query_rag(question: str) -> QueryResult:
    global _reranker

    if not _inited:
        _init(force_rebuild=False)

    my_text = (question or "").strip()
    if not my_text:
        return {"answer": "", "chunks": []}

    # bm25 + вектор
    retriever_results = _ensemble_retriever.invoke(my_text)

    if _reranker is None:
        _reranker = make_reranker()

    # контекст после реранка
    context, chunks = build_context_after_rerank(my_text, retriever_results, _reranker, top_k=5)
    answer = chat_answer(my_text, context)
    return {"answer": answer, "chunks": chunks}


def get_answer(question: str) -> str:
    return query_rag(question)["answer"]
