import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Загрузить модели для молока
try:
    model_udoy = joblib.load('model_udoy.joblib')
    model_zhir = joblib.load('model_zhir.joblib')
    model_belok = joblib.load('model_belok.joblib')
except FileNotFoundError as e:
    st.error(f"Ошибка загрузки моделей молока: {e}")
    st.stop()

# Загрузить модели для крови
blood_models = {}
short_names = {
    'Аланинаминотрансфераза, МЕ/л': 'alt',
    'Аспартатаминотрансфераза, МЕ/л': 'ast',
    'Гаммаглутамилтрансфераза, МЕ/л': 'ggt',
    'Щёлочная фосфатаза, МЕ/л': 'alp',
    'Холестерин, ммоль/л': 'chol',
    'Триглицериды, ммоль/л': 'trig',
    'Билирубин общий, мкмоль/л': 'bili',
    'Креатинин, мкмоль/л': 'crea',
    'Мочевина, ммоль/л': 'urea',
    'Липаза, МЕ/л': 'lipa',
    'Креатинкиназа, МЕ/л': 'ck',
    'Общий белок, г/л': 'tp',
    'Альбумин, г/л': 'alb',
    'Глобулин, г/л -': 'glob',
    'Натрий, ммоль/л': 'na',
    'Калий, ммоль/л': 'k',
    'Кальций, ммоль/л': 'ca',
    'Фосфор, ммоль/л': 'p',
    'Хлориды, ммоль/л': 'cl',
    'Глутаматдегидрогеназа, МЕ/л': 'gldh'
}
for param, short in short_names.items():
    try:
        blood_models[param] = joblib.load(f'model_blood_{short}.joblib')
    except FileNotFoundError:
        st.error(f"Модель для {param} не найдена. Убедитесь, что файл model_blood_{short}.joblib существует.")
        blood_models[param] = None

# Создать интерфейс
st.title('Предсказание изменений параметров после ввода препарата')

# Раздел для молока
st.header('Предсказание для молока')
udoy = st.number_input('Текущий суточный удой, кг', min_value=0.0, max_value=100.0, value=30.0, key='udoy')
zhir = st.number_input('Текущий жир, %', min_value=0.0, max_value=100.0, value=3.8, key='zhir')
belok = st.number_input('Текущий белок, %', min_value=0.0, max_value=100.0, value=3.3, key='belok')
group = st.number_input('Группа', min_value=1, max_value=6, value=1, key='group')

# Раздел для крови
with st.expander('Предсказание для параметров крови (введите текущие значения)'):
    blood_inputs = {}
    for param in blood_models.keys():
        blood_inputs[param] = st.number_input(f'Текущий {param}', min_value=0.0, max_value=1000.0, value=0.0, key=param)

# Кнопка предсказания
if st.button('Предсказать'):
    # Предсказание для молока
    milk_results = []
    input_milk = np.array([[udoy, zhir, belok, group]])
    try:
        delta_udoy = model_udoy.predict(input_milk)[0]
        delta_zhir = model_zhir.predict(input_milk)[0]
        delta_belok = model_belok.predict(input_milk)[0]
        new_udoy = udoy + delta_udoy
        new_zhir = zhir + delta_zhir
        new_belok = belok + delta_belok
        
        milk_results = [
            {'Параметр': 'Суточный удой, кг', 'Текущее значение': f'{udoy:.2f}', 'Новое значение': f'{new_udoy:.2f}', 'Изменение': f'{delta_udoy:.2f}'},
            {'Параметр': 'Жир, %', 'Текущее значение': f'{zhir:.2f}', 'Новое значение': f'{new_zhir:.2f}', 'Изменение': f'{delta_zhir:.2f}'},
            {'Параметр': 'Белок, %', 'Текущее значение': f'{belok:.2f}', 'Новое значение': f'{new_belok:.2f}', 'Изменение': f'{delta_belok:.2f}'}
        ]
        
        st.subheader('Результаты для молока')
        df_milk = pd.DataFrame(milk_results)
        st.table(df_milk)
    except Exception as e:
        st.error(f"Ошибка предсказания для молока: {e}")

    # Предсказание для крови
    blood_results = []
    for param, model in blood_models.items():
        if model is None:
            continue
        current = blood_inputs[param]
        if current == 0.0:
            continue
        input_blood = np.array([[current, group]])
        try:
            delta = model.predict(input_blood)[0]
            new_value = current + delta
            blood_results.append({
                'Параметр': param,
                'Текущее значение': f'{current:.2f}',
                'Новое значение': f'{new_value:.2f}',
                'Изменение': f'{delta:.2f}'
            })
        except Exception as e:
            st.error(f"Ошибка предсказания для {param}: {e}")
    
    if blood_results:
        st.subheader('Результаты для крови')
        df_blood = pd.DataFrame(blood_results)
        st.table(df_blood)
    else:
        st.write('Введите ненулевые значения для параметров крови для предсказания.')