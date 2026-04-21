from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


def make_reranker():
    return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


def build_context_after_rerank(my_text: str, retriever_results: list[Document], reranker, top_k: int = 5):
    # пары (запрос, текст чанка)
    pairs = [[my_text, doc.page_content] for doc in retriever_results]
    rerank_scores = reranker.predict(pairs)

    # сортировка по убыванию скора
    sorted_indices = sorted(
        range(len(rerank_scores)),
        key=lambda i: rerank_scores[i],
        reverse=True,
    )
    reranked_docs = [retriever_results[i] for i in sorted_indices[:top_k]]

    # одна строка контекста для llm
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    return context
