import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузить модели
model_udoy = joblib.load('model_udoy.joblib')
model_zhir = joblib.load('model_zhir.joblib')
model_belok = joblib.load('model_belok.joblib')

# Настройка страницы
st.set_page_config(page_title='Предсказание изменений параметров молока', layout='wide')

# Заголовок
st.title('Предсказание изменений параметров молока после ввода препарата')

# Ввод параметров
st.sidebar.header('Введите текущие параметры:')
удой = st.sidebar.number_input('Суточный удой, кг', min_value=0.0, max_value=100.0, value=30.0)
жир = st.sidebar.number_input('Жир, %', min_value=0.0, max_value=100.0, value=3.8)
белок = st.sidebar.number_input('Белок, %', min_value=0.0, max_value=100.0, value=3.3)
группа = st.sidebar.selectbox('Группа', [1, 2, 3])

# Предсказать
if st.sidebar.button('Предсказать'):
    # Подготовить данные для предсказания
    input_data = np.array([[удой, жир, белок, группа]])
    
    # Предсказать изменения
    delta_udoy = model_udoy.predict(input_data)[0]
    delta_zhir = model_zhir.predict(input_data)[0]
    delta_belok = model_belok.predict(input_data)[0]
    
    # Рассчитать новые значения
    new_udoy = удой + delta_udoy
    new_zhir = жир + delta_zhir
    new_belok = белок + delta_belok
    
    # Вывести результаты в основной части страницы
    st.header('Результаты предсказания:')
    
    # Таблица с данными
    data = {
        'Параметр': ['Суточный удой', 'Жир', 'Белок'],
        'Текущее значение': [удой, жир, белок],
        'Предсказанное изменение': [delta_udoy, delta_zhir, delta_belok],
        'Новое значение': [new_udoy, new_zhir, new_belok]
    }
    df_results = pd.DataFrame(data)
    st.dataframe(df_results.style.highlight_max(axis=0))
    
    # Графики
    st.subheader('Графическое представление:')
    
    # Гистограмма изменений
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Параметр', y='Предсказанное изменение', data=df_results, ax=ax)
    ax.set_title('Предсказанные изменения параметров')
    ax.set_ylabel('Изменение')
    st.pyplot(fig)
    
    # Линейный график текущих и новых значений
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=['Текущее', 'Новое'], y=[удой, new_udoy], label='Удой', marker='o', ax=ax2)
    sns.lineplot(x=['Текущее', 'Новое'], y=[жир, new_zhir], label='Жир', marker='o', ax=ax2)
    sns.lineplot(x=['Текущее', 'Новое'], y=[белок, new_belok], label='Белок', marker='o', ax=ax2)
    ax2.set_title('Изменение параметров до и после ввода препарата')
    ax2.set_ylabel('Значение')
    st.pyplot(fig2)
