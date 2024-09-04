import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

def create_interaction_features(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    if 'Inflight wifi service' in X.columns and 'Inflight entertainment' in X.columns:
        X['wifi_entertainment'] = X['Inflight wifi service'] * X['Inflight entertainment']
    
    service_columns = ['Food and drink', 'Inflight service', 'On-board service']
    if all(col in X.columns for col in service_columns):
        X['total_service'] = X[service_columns].sum(axis=1)
    
    return X

@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'best_model.joblib')
    return joblib.load(model_path)

def prepare_input(input_data):
    expected_columns = [
        'Age', 'Flight Distance', 'Inflight wifi service',
        'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding',
        'Seat comfort', 'Inflight entertainment', 'On-board service',
        'Leg room service', 'Baggage handling', 'Checkin service',
        'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
        'Arrival Delay in Minutes'
    ]
    
    df = pd.DataFrame([input_data])
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    
    # Crear características adicionales
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['service_rating'] = df[['Inflight wifi service', 'Food and drink', 'Inflight entertainment', 'On-board service']].mean(axis=1)
    df['convenience_rating'] = df[['Online boarding', 'Gate location', 'Ease of Online booking']].mean(axis=1)
    
    return df

def plot_shap_summary(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig = shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(fig)

st.title('Predicción de Satisfacción de Aerolíneas')

# Cargar el modelo
model = load_model()

# Mostrar los pasos del pipeline del modelo
st.write("Pasos del pipeline del modelo:")
for step in model.steps:
    st.write(f"- {step[0]}: {type(step[1]).__name__}")

# Crear los inputs para el usuario
age = st.slider('Edad', 0, 100, 30)
flight_distance = st.number_input('Distancia de vuelo', min_value=0, value=1000)
wifi_service = st.slider('Servicio de WiFi a bordo', 0, 5, 3)
departure_arrival_time = st.slider('Conveniencia de hora de salida/llegada', 0, 5, 3)
ease_of_booking = st.slider('Facilidad de reserva en línea', 0, 5, 3)
gate_location = st.slider('Ubicación de la puerta', 0, 5, 3)
food_and_drink = st.slider('Comida y bebida', 0, 5, 3)
online_boarding = st.slider('Embarque en línea', 0, 5, 3)
seat_comfort = st.slider('Comodidad del asiento', 0, 5, 3)
inflight_entertainment = st.slider('Entretenimiento a bordo', 0, 5, 3)
onboard_service = st.slider('Servicio a bordo', 0, 5, 3)
leg_room_service = st.slider('Servicio de espacio para piernas', 0, 5, 3)
baggage_handling = st.slider('Manejo de equipaje', 0, 5, 3)
checkin_service = st.slider('Servicio de check-in', 0, 5, 3)
inflight_service = st.slider('Servicio en vuelo', 0, 5, 3)
cleanliness = st.slider('Limpieza', 0, 5, 3)
departure_delay = st.number_input('Retraso en la salida (minutos)', min_value=0, value=0)
arrival_delay = st.number_input('Retraso en la llegada (minutos)', min_value=0, value=0)

if st.button('Predecir Satisfacción'):
    input_data = {
        'Age': age,
        'Flight Distance': flight_distance,
        'Inflight wifi service': wifi_service,
        'Departure/Arrival time convenient': departure_arrival_time,
        'Ease of Online booking': ease_of_booking,
        'Gate location': gate_location,
        'Food and drink': food_and_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': inflight_entertainment,
        'On-board service': onboard_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Departure Delay in Minutes': departure_delay,
        'Arrival Delay in Minutes': arrival_delay
    }
    
    prepared_input = prepare_input(input_data)
    st.write("Características preparadas:", prepared_input.columns.tolist())
    prediction = model.predict(prepared_input)
    probabilities = model.predict_proba(prepared_input)
    
    if prediction[0] == 1:
        st.success(f'El pasajero probablemente estará satisfecho con una probabilidad del {probabilities[0][1]:.2%}')
    else:
        st.error(f'El pasajero probablemente estará insatisfecho con una probabilidad del {probabilities[0][0]:.2%}')
    
    if st.button('Mostrar explicación del modelo'):
        plot_shap_summary(model, prepared_input)

    # Sistema de feedback
    feedback = st.radio("¿Fue útil esta predicción?", ("Sí", "No"))
    if st.button('Enviar feedback'):
        # Aquí puedes implementar la lógica para guardar el feedback
        st.write("Gracias por tu feedback!")

st.sidebar.header('Acerca de')
st.sidebar.info('Esta aplicación predice la satisfacción de los pasajeros de aerolíneas basándose en varios factores.')

# Mostrar otras visualizaciones guardadas
st.image('reports/figures/confusion_matrix.png')
st.image('reports/figures/roc_curve.png')