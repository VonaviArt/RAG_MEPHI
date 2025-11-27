from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import lancedb as ld
import pyarrow as pa
from langchain_huggingface import HuggingFaceEmbeddings
import os


file_path = r"C:\Users\Lenovo\Desktop\RAG\RAG_инфа по распорядку и пр.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
# for doc in docs:
#     print(doc.page_content) #грузанули в docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
all_texts = []#все страницы отчанковали
for doc in docs:
    chunks = text_splitter.split_text(doc.page_content)
    all_texts.extend(chunks)


#эмбеддинги прям по видео, но есть кучи моделек на русском
hf_embeddings_model = HuggingFaceEmbeddings(model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": "cpu"})
embeddings = hf_embeddings_model.embed_documents(all_texts)

#оки-доки, теперь вбд
db = ld.connect(r"C:\Users\Lenovo\Desktop\RAG\db")
schema = pa.schema([ #lance не принимает схемы без pyarrow потому что мать - апаче
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("embedding", pa.list_(pa.float32(), 768))
])
try:
    table = db.open_table("documents")
except Exception:
    table = db.create_table("documents", schema=schema)
# вставляем данные
records = []
for i in range(len(all_texts)):
    record = {
        "id": i,
        "text": all_texts[i],
        "embedding": embeddings[i]
    }
    records.append(record)
table.add(records) # и добавим в таблицу
my_text = "Научный читальный зал (НЧЗ) Понедельник - пятница со скольки до скольки работает"# У - уникальность
my_text_embedded = hf_embeddings_model.embed_query(my_text) #перевели в эмбеддинг запрос\
# найдем похожие эмбеддинги в таблице
results = table.search(
    my_text_embedded
)

# собираем все чанки в один контекст
results_list = results.to_list()
context = "\n\n".join(record["text"] for record in results_list)
#апишка к чату гпт
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key='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',#ключ кринжово хранить в коде, потом подумаю что делать
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "system",
            "content": "ты - помощник Игорь. ты заикаешся и занудствуешь. представься в начале. используй искючительно контекст для ответа."
        },
        {
            "role": "user",
            "content": f"используя следующий контекст, ответь на вопрос:\n\n{context}\n\nВопрос: {my_text}"
        }
    ],
)

print(completion.choices[0].message.content)

