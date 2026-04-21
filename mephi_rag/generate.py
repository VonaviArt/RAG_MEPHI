# генерация ответа по контексту

from ollama import chat


def chat_answer(my_text: str, context: str) -> str:
    # локальная модель, системный и пользовательский промпт
    response = chat(
        model="qwen3:4b-thinking",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты - помощник студентам МИФИ. "
                    "Отвечай только на основе контекста. "
                    "Если данных достаточно — ответь кратко и по делу."
                    "Если данных нет - напиши 'Информация отсутствует в документе'."
                    "Не добавляй фразу об отсутствии информации, если ты уже что-то сказал."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Контекст:\n{context}\n\n"
                    f"Вопрос: {my_text}"
                ),
            },
        ],
    )

    return response.message.content or ""
