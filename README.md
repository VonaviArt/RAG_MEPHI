![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

# RAG-ассистент для студентов МИФИ

## Помощник для поиска информации в документах МИФИ на основе технологии RAG (Retrieval-Augmented Generation).

## 🔄 RAG Pipeline

| Step | Component | Description |
|:---:|:---|:---|
| 1 | **Query** | Пользовательский запрос |
| 2 | **EnsembleRetriever** | Dense (эмбеддинги) + BM25 (ключевые слова) |
| 3 | **Reranker** | Cross-Encoder (`LaBSE-en-ru`) |
| 4 | **Context** | Топ-5 наиболее релевантных чанков |
| 5 | **LLM** | Ollama (`T-lite 2.1`) |
| 6 | **Answer** | Финальный ответ |

---

## Примеры работы:
<img width="700" height="750" alt="image" src="https://github.com/user-attachments/assets/51993283-fe57-4ec1-a6e5-44c5d42ae2b7" /> 
<img width="700" height="750" alt="image" src="https://github.com/user-attachments/assets/1a7bb83b-ef31-4f46-a842-ffb3c155ec63" />

ʕ ᵔᴥᵔ ʔ
