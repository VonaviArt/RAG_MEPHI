![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![RAGAS](https://img.shields.io/badge/RAGAS-0.2.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

# RAG ассистент для помощи студентам с поиском информации в документах МИФИ.
```markdown
## 🔄 RAG Pipeline

```mermaid
graph LR
    Q["User Query"] --> R["Hybrid Retriever<br/>(Dense + BM25)"]
    R --> RR["Cross-Encoder<br/>Reranker"]
    RR --> C["Top-5 Context"]
    C --> L["LLM on Ollama<br/>(Llama 3)"]
    L --> A["Final Answer"]
    
    style R fill:#f9f,stroke:#333,stroke-width:1px
    style RR fill:#bbf,stroke:#333,stroke-width:1px
    style L fill:#bfb,stroke:#333,stroke-width:1px

  
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
