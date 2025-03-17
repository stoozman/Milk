import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузить модели
model_udoy = joblib.load('model_udoy.joblib')
model_zhir = joblib.load('model_zhir.joblib')
model_belok = joblib.load('model_belok.joblib')

# Создать интерфейс
st.title('Предсказание изменений параметров после ввода препарата')
st.write('Введите текущие параметры для предсказания изменений:')

# Ввод параметров
удой = st.number_input('Текущий суточный удой, кг', min_value=0.0, max_value=100.0, value=30.0)
жир = st.number_input('Текущий жир, %', min_value=0.0, max_value=100.0, value=3.8)
белок = st.number_input('Текущий белок, %', min_value=0.0, max_value=100.0, value=3.3)
группа = st.number_input('Группа', min_value=1, max_value=3, value=1)

# Предсказать
if st.button('Предсказать'):
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
    
    # Вывести результаты
    st.write('Предсказанные изменения:')
    st.write(f'Изменение удоя: {delta_udoy:.2f} кг')
    st.write(f'Изменение жира: {delta_zhir:.2f} %')
    st.write(f'Изменение белка: {delta_belok:.2f} %')
    
    st.write('Новые значения после ввода препарата:')
    st.write(f'Новый удой: {new_udoy:.2f} кг')
    st.write(f'Новый жир: {new_zhir:.2f} %')
    st.write(f'Новый белок: {new_belok:.2f} %')