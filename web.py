import streamlit as st
from RAG import get_answer

def answer_question(question: str) -> str:
    return get_answer(question)


st.title("Вопросно-ответная форма")
st.write("Введите вопрос в поле ниже.")

# uploaded_file = st.file_uploader("Загрузить файл", type=["pdf", "docx", "txt"])
# if uploaded_file is not None:
#     st.success(f"Файл загружен: {uploaded_file.name}")

user_question = st.text_input("Ваш вопрос:")

# при отправке
if st.button("Получить ответ"):
    if user_question.strip() == "":
        st.warning("Введите вопрос.")
    else:
        try:
            with st.spinner("Обрабатываю..."):
                answer = answer_question(user_question)
            st.success("Готово!")
            st.write("### Ответ:")
            st.write(answer)
        except Exception as e:
            st.error(f"Произошла ошибка при обработке запроса: {str(e)}")
            st.info("Пожалуйста, убедитесь, что RAG система инициализирована корректно.")



