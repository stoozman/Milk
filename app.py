import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
        'Новое значение': [new_udoy, new_zhir, new_belok]
    }
    df_results = pd.DataFrame(data)
    st.dataframe(df_results)
