import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

st.title("Анализ данных и модель")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите predictive_maintenance.csv из папки data/", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Проверка и удаление столбцов
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    if existing_columns:
        data = data.drop(columns=existing_columns)
    else:
        st.warning("Некоторые ожидаемые столбцы отсутствуют в загруженном файле. Проверьте, что загружаете predictive_maintenance.csv.")

    # Проверка наличия обязательных столбцов
    required_columns = ['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear', 'Machine failure']
    if not all(col in data.columns for col in required_columns):
        st.error("Файл не содержит всех необходимых столбцов. Ожидаются: Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Machine failure.")
    else:
        # Предобработка
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
        numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Разделение данных
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Сохранение модели
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_FILE = os.path.join(SCRIPT_DIR, 'model.pkl')
        joblib.dump(model, MODEL_FILE)

        # Оценка
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для ROC-AUC
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Визуализация результатов
        st.header("Результаты")
        st.write(f"Точность модели: {accuracy:.2f}")
        st.write(f"ROC-AUC: {roc_auc:.2f}")
        st.subheader("Матрица ошибок")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # ROC-кривая
        st.subheader("ROC-кривая")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайное угадывание')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC-кривая')
        ax.legend()
        st.pyplot(fig)

        # Форма для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            type_input = st.selectbox("Type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]", value=300.0)
            process_temp = st.number_input("Process temperature [K]", value=310.0)
            rotational_speed = st.number_input("Rotational speed [rpm]", value=1500)
            torque = st.number_input("Torque [Nm]", value=40.0)
            tool_wear = st.number_input("Tool wear [min]", value=0)
            submit_button = st.form_submit_button("Предсказать")

            if submit_button:
                # Преобразование введенных данных
                input_data = pd.DataFrame({
                    'Type': [LabelEncoder().fit(['L', 'M', 'H']).transform([type_input])[0]],
                    'Air temperature': [air_temp],
                    'Process temperature': [process_temp],
                    'Rotational speed': [rotational_speed],
                    'Torque': [torque],
                    'Tool wear': [tool_wear]
                })
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]
                st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")