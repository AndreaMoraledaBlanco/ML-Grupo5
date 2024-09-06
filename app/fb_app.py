import streamlit as st
import pandas as pd
import joblib
import csv
from datetime import datetime
import os

# Cargar el modelo XGBoost
model = joblib.load('../src/Modelos/xgboost_model.joblib')

# Función para hacer predicciones
def predict(data):
    prediction = model.predict(data)
    return prediction[0]

# Función para guardar los resultados en CSV
def save_to_csv(data, prediction, user_feedback):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = '../src/Data/feedback_results.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp'] + list(data.keys()) + ['pred_satisfaction', 'feedback_satisfaction'])
        writer.writerow([timestamp] + list(data.values()) + [
            1 if prediction == 1 else 0,
            1 if user_feedback == 'Satisfecho' else 0
        ])

# Crear la aplicación Streamlit
st.title('Feedback de Satisfacción del Cliente')

# Crear el formulario
st.header('Ingrese los datos del cliente:')

gender = st.selectbox('Género', ['Masculino', 'Femenino'])
customer_type = st.selectbox('Tipo de Cliente', ['Leal', 'Desleal'])
age = st.number_input('Edad', min_value=0, max_value=120)
type_of_travel = st.selectbox('Tipo de Viaje', ['Personal', 'Negocios'])
class_type = st.selectbox('Clase', ['Económica', 'Económica Plus', 'Negocios'])
flight_distance = st.number_input('Distancia de Vuelo', min_value=0)
inflight_wifi_service = st.slider('Servicio de WiFi a bordo', 0, 5)
departure_arrival_time_convenient = st.slider('Conveniencia del horario de salida/llegada', 0, 5)
ease_of_online_booking = st.slider('Facilidad de reserva en línea', 0, 5)
gate_location = st.slider('Ubicación de la puerta', 0, 5)
food_and_drink = st.slider('Comida y bebida', 0, 5)
online_boarding = st.slider('Embarque en línea', 0, 5)
seat_comfort = st.slider('Confort del asiento', 0, 5)
inflight_entertainment = st.slider('Entretenimiento a bordo', 0, 5)
onboard_service = st.slider('Servicio a bordo', 0, 5)
leg_room_service = st.slider('Espacio para las piernas', 0, 5)
baggage_handling = st.slider('Manejo de equipaje', 0, 5)
checkin_service = st.slider('Servicio de check-in', 0, 5)
inflight_service = st.slider('Servicio durante el vuelo', 0, 5)
cleanliness = st.slider('Limpieza', 0, 5)
departure_delay = st.number_input('Retraso en la salida (minutos)', min_value=0)
arrival_delay = st.number_input('Retraso en la llegada (minutos)', min_value=0)

# Nueva pregunta sobre la satisfacción general con la aerolínea
satisfaction = st.radio(
    "En general, ¿cómo se encuentra con el servicio de la aerolínea?",
    ('Satisfecho', 'Insatisfecho')
)

# Inicializar variables de estado
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Botón para hacer la predicción
if st.button('Mostrar predición'):
    # Preparar los datos para la predicción
    data = {
        'Gender': 0 if gender == 'Masculino' else 1,
        'Customer Type': 0 if customer_type == 'Leal' else 1,
        'Age': age,
        'Type of Travel': 0 if type_of_travel == 'Personal' else 1,
        'Class': 0 if class_type == 'Económica' else (1 if class_type == 'Económica Plus' else 2),
        'Flight Distance': flight_distance,
        'Inflight wifi service': inflight_wifi_service,
        'Departure/Arrival time convenient': departure_arrival_time_convenient,
        'Ease of Online booking': ease_of_online_booking,
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
        'Arrival Delay in Minutes': arrival_delay,
    }

    # Convertir los datos a DataFrame
    data_df = pd.DataFrame([data])

    # Hacer la predicción
    prediction = predict(data_df)

    # Guardar la predicción y los datos en el estado de la sesión
    st.session_state.prediction_made = True
    st.session_state.prediction = prediction
    st.session_state.data = data

# Mostrar el resultado si se ha hecho una predicción
if st.session_state.prediction_made:
    st.write('Predicción de satisfacción:', 'Satisfecho' if st.session_state.prediction == 1 else 'Insatisfecho')

    # Botón para guardar los resultados
    if st.button('Enviar Formulario'):
        save_to_csv(st.session_state.data, st.session_state.prediction, satisfaction)
        st.success(f'Resultados guardados exitosamente. Su opinión sobre la predicción: {satisfaction}')

        # Reiniciar el formulario
        st.session_state.prediction_made = False
        st.session_state.prediction = None
        st.session_state.data = None
        
        # Notificar al usuario que los valores han sido reiniciados
        st.info("Formulario reiniciado. Ingrese nuevos datos para realizar otra predicción.")


