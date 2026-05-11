![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

# RAG-ассистент для студентов МИФИ

Помощник для поиска информации в документах МИФИ на основе технологии RAG (Retrieval-Augmented Generation).

## 🔄 RAG Pipeline

| Step | Component | Description |
|:---:|:---|:---|
| 1 | **Query** | Пользовательский запрос |
| 2 | **EnsembleRetriever** | Dense (эмбеддинги) + BM25 (ключевые слова) |
| 3 | **Reranker** | Cross-Encoder (`LaBSE-en-ru`) |
| 4 | **Context** | Топ-5 наиболее релевантных чанков |
| 5 | **LLM** | Ollama (`T-lite 2.1`) |
| 6 | **Answer** | Финальный ответ |

**Pipeline flow:** `Query → EnsembleRetriever → Reranker → Context → Ollama → Answer`

---

## Результаты эксперимента

| Метрика | Baseline | +Reranker | Δ | Δ% |
|:---|:---:|:---:|:---:|:---:|
| **Context Precision** | 0.55 | **0.76** | **+0.21** | **+38%** |
| Faithfulness | 0.76 | 0.84 | +0.08 | +11% |
| Answer Relevancy | 0.80 | 0.84 | +0.04 | +5% |
| Context Recall | 0.66 | 0.69 | +0.03 | +5% |

> **Ключевой результат:** Context Precision выросла на **38%** после добавления реранкера.

---

## 🎯 Проблема

Без реранкера **45% контекста** — нерелевантный шум, из-за чего LLM путается и даёт неточные ответы.

## 💡 Решение

Добавлен **Cross-Encoder реранкер** (`BAAI/bge-reranker-v2-m3`), который отсеивает мусорные чанки перед подачей в LLM.

## ✨ Результаты улучшения

| Показатель | Без реранкера | С реранкером | Улучшение |
|:---|:---:|:---:|:---:|
| Context Precision | 0.55 | 0.76 | **+38%** |
| Faithfulness | 0.76 | 0.84 | **+11%** |

---

## Примеры работы:
<img width="700" height="750" alt="image" src="https://github.com/user-attachments/assets/51993283-fe57-4ec1-a6e5-44c5d42ae2b7" /> 
<img width="700" height="750" alt="image" src="https://github.com/user-attachments/assets/1a7bb83b-ef31-4f46-a842-ffb3c155ec63" />

ʕ ᵔᴥᵔ ʔ
