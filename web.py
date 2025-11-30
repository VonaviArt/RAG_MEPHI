import streamlit as st

def answer_question(question: str) -> str:
    return f"Ответ на вопрос: {question}\n\n"


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
        with st.spinner("Обрабатываю..."):
            answer = answer_question(user_question)
        st.success("Готово!")
        st.write("### Ответ:")
        st.write(answer)



