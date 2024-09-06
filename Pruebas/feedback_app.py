import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb

# Definir el archivo CSV donde se almacenará el feedback
FEEDBACK_CSV = 'feedback.csv'

# Cargar el modelo XGBoost entrenado
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')  # Cargar el modelo guardado

# Función para cargar feedback existente
def load_feedback():
    try:
        return pd.read_csv(FEEDBACK_CSV)
    except FileNotFoundError:
        return pd.DataFrame(columns=['timestamp', 'true_label', 'predicted_label', 'feedback_score'])

# Función para guardar el feedback en el archivo CSV
def save_feedback(feedback_data):
    feedback_data.to_csv(FEEDBACK_CSV, index=False)

# Función para agregar feedback al DataFrame existente
def add_feedback(true_label, predicted_label, feedback_score):
    feedback_data = load_feedback()
    # Crear un nuevo DataFrame con los nuevos datos de feedback
    new_feedback = pd.DataFrame({
        'timestamp': [datetime.now().isoformat()],
        'true_label': [true_label],
        'predicted_label': [predicted_label],
        'feedback_score': [feedback_score]
    })
    # Concatenar los DataFrames
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)
    save_feedback(feedback_data)
    return feedback_data

# Función para predecir usando el modelo XGBoost
def predict_satisfaction(input_data):
    # Convertir los datos de entrada en un DataFrame en el mismo orden que el modelo espera
    input_df = pd.DataFrame([input_data])
    # Realizar la predicción
    prediction = model.predict(input_df)
    return int(prediction[0])  # Devolver la predicción como un entero

# Título de la aplicación
st.title('Sistema de Recogida de Feedback para Satisfacción del Cliente de Aerolíneas')

# Sección de formulario para recolección de feedback
st.header('Formulario de Feedback')
true_label = st.selectbox('Satisfacción General (1-5):', [1, 2, 3, 4, 5])
feedback_score = st.selectbox('¿Recomendaría la aerolínea? (1-5):', [1, 2, 3, 4, 5])

# Crear los datos de entrada para el modelo de predicción
st.subheader('Ingrese los datos del cliente para la predicción')
input_data = {
    'Gender': st.selectbox('Género (0=Mujer, 1=Hombre):', [0, 1]),
    'Customer Type': st.selectbox('Tipo de Cliente (0=Leal, 1=No leal):', [0, 1]),
    'Age': st.slider('Edad:', 1, 100),
    'Type of Travel': st.selectbox('Tipo de Viaje (0=Personal, 1=Negocios):', [0, 1]),
    'Class': st.selectbox('Clase (0=Económica, 1=Business, 2=Primera Clase):', [0, 1, 2]),
    'Flight Distance': st.slider('Distancia del Vuelo:', 0, 10000),
    'Inflight wifi service': st.slider('Servicio de wifi en vuelo (0-5):', 0, 5),
    'Departure/Arrival time convenient': st.slider('Comodidad del tiempo de salida/llegada (0-5):', 0, 5),
    'Ease of Online booking': st.slider('Facilidad de reserva online (0-5):', 0, 5),
    'Gate location': st.slider('Ubicación de la puerta (0-5):', 0, 5),
    'Food and drink': st.slider('Comida y bebida (0-5):', 0, 5),
    'Online boarding': st.slider('Embarque online (0-5):', 0, 5),
    'Seat comfort': st.slider('Comodidad del asiento (0-5):', 0, 5),
    'Inflight entertainment': st.slider('Entretenimiento en vuelo (0-5):', 0, 5),
    'On-board service': st.slider('Servicio a bordo (0-5):', 0, 5),
    'Leg room service': st.slider('Servicio de espacio para las piernas (0-5):', 0, 5),
    'Baggage handling': st.slider('Manejo del equipaje (0-5):', 0, 5),
    'Checkin service': st.slider('Servicio de check-in (0-5):', 0, 5),
    'Inflight service': st.slider('Servicio en vuelo (0-5):', 0, 5),
    'Cleanliness': st.slider('Limpieza (0-5):', 0, 5),
    'Departure Delay in Minutes': st.slider('Retraso de salida en minutos:', 0, 2000),
    'Arrival Delay in Minutes': st.slider('Retraso de llegada en minutos:', 0, 2000),
}

# Botón para enviar feedback
if st.button('Enviar Feedback'):
    # Predicción usando el modelo XGBoost
    predicted_label = predict_satisfaction(input_data)
    feedback_data = add_feedback(true_label, predicted_label, feedback_score)
    st.success('¡Gracias por su feedback!')
    st.write('Feedback registrado con éxito.')
    st.write(feedback_data.tail())  # Mostrar los últimos datos de feedback para confirmación

# Sección para visualizar el feedback almacenado
st.header('Visualización de Feedback Recopilado')

# Mostrar feedback almacenado
feedback_data = load_feedback()
if not feedback_data.empty:
    st.write(feedback_data)

    # Gráficos de métricas de rendimiento
    st.subheader('Métricas de Rendimiento del Modelo Basadas en Feedback')
    
    # Gráfica de conteo de feedback de satisfacción
    st.bar_chart(feedback_data['true_label'].value_counts(), use_container_width=True)

    # Gráfica de precisión del modelo en base a feedback
    feedback_data['correct_prediction'] = feedback_data['true_label'] == feedback_data['predicted_label']
    accuracy = feedback_data['correct_prediction'].mean()
    st.write(f'Precisión del modelo basado en el feedback: {accuracy:.2f}')
else:
    st.write('No hay feedback recopilado todavía.')


