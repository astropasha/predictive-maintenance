import streamlit as st

st.set_page_config(page_title="Предиктивное обслуживание", layout="wide")

# Определение страниц
pg = st.navigation([
    st.Page("pages/analysis_and_model.py", title="Анализ и модель"),
    st.Page("pages/presentation.py", title="Презентация")
])

# Запуск страницы
pg.run()