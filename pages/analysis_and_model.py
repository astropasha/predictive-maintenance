import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC  # Добавляем SVM
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
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

        # Определение моделей
        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "SVM": SVC(probability=True, random_state=42)  # Добавляем SVM с probability=True для ROC-AUC
        }

        # Обучение и сравнение моделей
        st.header("Обучение и сравнение моделей")
        if st.button("Обучить и сравнить модели"):
            results = {}
            best_model_name = None
            best_accuracy = 0

            for name, model in models.items():
                # Обучение модели
                model.fit(X_train, y_train)
                # Предсказания на тестовой выборке
                test_predictions = model.predict(X_test)
                test_acc = accuracy_score(y_test, test_predictions)
                test_cm = confusion_matrix(y_test, test_predictions)
                test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                test_classification_report = classification_report(y_test, test_predictions, output_dict=True)

                results[name] = {
                    "model": model,
                    "accuracy": test_acc,
                    "confusion_matrix": test_cm,
                    "roc_auc": test_roc_auc,
                    "classification_report": test_classification_report
                }

                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_model_name = name

            # Сохранение наилучшей модели
            best_model = results[best_model_name]["model"]
            st.session_state["model"] = best_model
            st.session_state["model_name"] = best_model_name

            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            MODEL_FILE = os.path.join(SCRIPT_DIR, 'model.pkl')
            joblib.dump(best_model, MODEL_FILE)

            # Вывод результатов
            st.write("### Результаты сравнения моделей:")
            for name, metrics in results.items():
                st.write(f"**{name}:**")
                st.write(f"Точность на тестовой выборке (Accuracy): {metrics['accuracy']:.2f}")
                st.write(f"ROC-AUC: {metrics['roc_auc']:.2f}")

                # Classification Report
                st.subheader(f"Classification Report для {name}")
                report_df = pd.DataFrame(metrics['classification_report']).transpose()
                st.write(report_df)

                # Confusion Matrix
                st.subheader(f"Матрица ошибок для {name}")
                fig, ax = plt.subplots()
                sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                # ROC-кривая
                st.subheader(f"ROC-кривая для {name}")
                fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.2f})")
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайное угадывание')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC-кривая для {name}')
                ax.legend()
                st.pyplot(fig)

            st.write(f"**Наилучшая модель: {best_model_name}** с точностью {best_accuracy:.2f}")

            # Точность на всей базе данных
            full_predictions = best_model.predict(X)
            full_acc = accuracy_score(y, full_predictions)
            st.write(f"Точность на всей базе данных (Accuracy): {full_acc:.2f}")

        # Форма для предсказания
        if "model" in st.session_state:
            st.header(f"Предсказание по новым данным (модель: {st.session_state['model_name']})")
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
                    input_data = pd.DataFrame({
                        'Type': [LabelEncoder().fit(['L', 'M', 'H']).transform([type_input])[0]],
                        'Air temperature': [air_temp],
                        'Process temperature': [process_temp],
                        'Rotational speed': [rotational_speed],
                        'Torque': [torque],
                        'Tool wear': [tool_wear]
                    })
                    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                    prediction = st.session_state["model"].predict(input_data)
                    prediction_proba = st.session_state["model"].predict_proba(input_data)[:, 1]
                    st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                    st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")