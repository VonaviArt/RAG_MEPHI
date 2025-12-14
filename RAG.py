from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import os
from langchain_classic.retrievers import EnsembleRetriever
from dotenv import load_dotenv

load_dotenv() 

# пдф
file_path = r"RAG_инфа по распорядку и пр.pdf"  # заменить позже
loader = PyPDFLoader(file_path)
docs = loader.load()  # docs = список по страницам

# чанкирование
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_texts = []
for doc in docs:
    # из каждой страницы делаем чанки
    chunks = text_splitter.split_text(doc.page_content)
    all_texts.extend(chunks)

# создаём объекты Document уже по чанкам (одна база для BM25 и векторов)
chunk_documents = [Document(page_content=text) for text in all_texts]

#эмбеддинги
hf_embeddings_model = HuggingFaceEmbeddings(
    model_name="cointegrated/LaBSE-en-ru",
    model_kwargs={"device": "cpu"}
)

# вбд (Chroma)
vector_db = Chroma.from_documents(  # vector_db = вся ВБД целиком
    documents=chunk_documents,
    embedding=hf_embeddings_model,
    persist_directory="./chroma_db",      # сохраняется на диск
    collection_name="university_docs"     # название коллекции
)

# ретриверы
# Векторный ретривер (Chroma)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# BM25 ретривер (лексический поиск) по тем же чанкам
bm25_retriever = BM25Retriever.from_documents(chunk_documents)
bm25_retriever.k = 5  # количество документов за раз рассмотрения

# Ансамблевый ретривер
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.2, 0.8]
)

# АПИ к ИИ
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("api_key_hf", "")
)


def get_answer(question: str) -> str:
    """
    Обрабатывает вопрос пользователя через RAG систему и возвращает ответ.
    
    Args:
        question: Вопрос пользователя
        
    Returns:
        Ответ модели на основе найденного контекста
    """
    # поиск и создание контекста
    retriever_results = ensemble_retriever.invoke(question)
    
    # контекст из найденных документов
    context = "\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in retriever_results
    )
    
    # print("Найденный контекст для запроса (ансамбль BM25 + векторы):")
    # print(context, "\n")  # \n для переноса строки
    
    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[
            {
                "role": "system",
                "content": "ты - помощник. Расскажи подробно о вопросе, опираясь на корректные данные. При отсутствии оных напиши об этом пользователю и посоветуй уточнить запрос или обратиться в деканат или учебный отдел"
            },
            {
                "role": "user",
                "content": (
                    f"используя следующий контекст, ответь на вопрос:\n\n"
                    f"{context}\n\n"
                    f"Вопрос: {question}"
                )
            }
        ],
    )
    
    # print("Ответ модели:")
    # print(completion.choices[0].message.content)
    
    return completion.choices[0].message.content