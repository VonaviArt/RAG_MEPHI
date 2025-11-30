from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# загрузка пдф
file_path = r"C:\Users\Lenovo\Desktop\RAG\RAG_инфа по распорядку и пр.pdf" #заменить позже
loader = PyPDFLoader(file_path)
docs = loader.load()

# разбиение текста на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
all_texts = []
for doc in docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_texts.extend(chunks)

# создание эмбеддингов
hf_embeddings_model = HuggingFaceEmbeddings(model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": "cpu"})
embeddings = hf_embeddings_model.embed_documents(all_texts)

# вбд
client = chromadb.Client()
collection = client.get_or_create_collection(name="documents")
ids = [str(i) for i in range(len(all_texts))]
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=all_texts
)

# эмбеддинг для запроса и поиск
my_text = "сколько стипендия у иностранцев троечников?"
my_text_embedded = hf_embeddings_model.embed_query(my_text)
results = collection.query(
    query_embeddings=[my_text_embedded],
    n_results=2
)

# ретривер
vector_store = Chroma(
    collection_name="documents",
    embedding_function=hf_embeddings_model,
)
retriever = vector_store.as_retriever()
retriever_results = vector_store.similarity_search_with_score(my_text)

# контекст
context = "\n\n".join(
    d.page_content if hasattr(d, 'page_content') else str(d)
    for d, score in retriever_results
)
print("Найденный контекст для запроса:")
print(context, "/n")

# Апи к ИИ
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=""# спрятать не в коде потом
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "system",
            "content": "ты - помощник Игорь. используй исключительно контекст для ответа."
        },
        {
            "role": "user",
            "content": f"используя следующий контекст, ответь на вопрос:\n\n{context}\n\nВопрос: {my_text}"
        }
    ],
)

print("Ответ модели:")
print(completion.choices[0].message.content)
