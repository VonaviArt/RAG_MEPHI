# эмбеддинги для векторного поиска в chroma

from langchain_huggingface import HuggingFaceEmbeddings


def hf_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name="cointegrated/LaBSE-en-ru",
        model_kwargs={"device": "cpu"},
    )
