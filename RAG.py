from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import os

HF_API_KEY = ""  # токен

# пдф
file_path = r"C:\Users\Lenovo\Desktop\3сем\RAG\RAG_итог\RAG_инфа по распорядку и пр.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# эмбеддинги
hf_embeddings_model = HuggingFaceEmbeddings(
    model_name="cointegrated/LaBSE-en-ru",
    model_kwargs={"device": "cpu"}
)

#хранение хромы
persist_directory = "./chroma_db"
collection_name = "university_docs"

# чанкирование
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_texts = []
for doc in docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_texts.extend(chunks)

chunk_documents = [Document(page_content=text) for text in all_texts]


# вбд (Chroma)
if os.path.exists(persist_directory):
    vector_db = Chroma(persist_directory="./chroma_db",
                       embedding_function=hf_embeddings_model)
    results = vector_db.get()
    chunk_documents = [Document(page_content=doc) for doc in results['documents']]
else:
    vector_db = Chroma.from_documents(chunk_documents,
                                      hf_embeddings_model,
                                      persist_directory="./chroma_db")# ретриверы
my_text = "сколько стипендия у студентов иностранцев?"

# векторный ретривер (Chroma)
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# BM25 ретривер (лексический поиск)
bm25_retriever = BM25Retriever.from_documents(chunk_documents)
bm25_retriever.k = 5

# ансамблевый ретривер
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.0, 1]
)

# поиск контекста
retriever_results = ensemble_retriever.invoke(my_text)

context = "\n\n".join(
    d.page_content if hasattr(d, "page_content") else str(d)
    for d in retriever_results
)

print("Найденный контекст для запроса (ансамбль BM25 + векторы):")
print(context[:1000] + "..." if len(context) > 1000 else context)
print()

#АПИ к ИИ
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "system",
            "content": (
                "Ты - помощник студентам МИФИ. "
                "Отвечай только на основе контекста. "
                "Если данных нет - напиши 'Информация отсутствует в документе'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Контекст:\n{context}\n\n"
                f"Вопрос: {my_text}"
            )
        }
    ],
)

answer = completion.choices[0].message.content

print("Ответ модели:")
print(answer)
