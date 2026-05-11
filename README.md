![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

# RAG ассистент для помощи студентам с поиском информации в документах МИФИ.

## 🔄 RAG Pipeline

1. **Query** → Пользовательский запрос
2. **Hybrid Retriever** → Поиск: Dense (эмбеддинги) + BM25 (ключевые слова)
3. **RRF Fusion** → Объединение результатов обоих поисков
4. **Reranker** → Cross-Encoder (BAAI/bge-reranker-v2-m3)
5. **Context** → Топ-5 наиболее релевантных чанков
6. **LLM** → Ollama (Llama 3 / Mistral)
7. **Answer** → Финальный ответ
  
| Метрика | Baseline | +Reranker | Δ | Δ% |
|:---|:---:|:---:|:---:|:---:|
| Context Precision | 0.55 | **0.76** | **+0.21** | **+38%** |
| Faithfulness | 0.76 | 0.84 | +0.08 | +11% |
| Answer Relevancy | 0.80 | 0.84 | +0.04 | +5% |
| Context Recall | 0.66 | 0.69 | +0.03 | +5% |

## 🎯 Проблема
Без реранкера 45% контекста — нерелевантный шум → LLM путается

## 💡 Решение
Добавили Cross-Encoder реранкер (BAAI/bge-reranker-v2-m3)

## ✨ Результат
- Context Precision: 0.55 → 0.76 (+38%)
- Faithfulness: 0.76 → 0.84 (+11%)



Докер контейнер

Примеры работы:


ʕ ᵔᴥᵔ ʔ
