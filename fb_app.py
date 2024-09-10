import streamlit as st
import pandas as pd
import joblib
import csv
from datetime import datetime
import os

# Función para cargar modelos de manera segura
def load_model(file_path):
    try:
        model = joblib.load(file_path)
        if isinstance(model, dict) and 'model' in model:
            return model['model']
        return model
    except FileNotFoundError:
        st.error(f"No se pudo encontrar el archivo del modelo: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo {file_path}: {str(e)}")
        return None

# Cargar los modelos
model_dir = 'src/Modelos'  # Ajusta esta ruta según la estructura de tu proyecto
xgboost_model = load_model(os.path.join(model_dir, 'xgboost_model.joblib'))
logistic_model = load_model(os.path.join(model_dir, 'logistic_model.joblib'))
stack_model = load_model(os.path.join(model_dir, 'stack_model.joblib'))
neuronal_model = load_model(os.path.join(model_dir, 'neuronal.joblib'))

# Función para hacer predicciones
def predict(data, model):
    if model is not None:
        try:
            pred = model.predict(data)[0]
            return pred
        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")
    return None

# Función para guardar los resultados en CSV
def save_to_csv(data, predictions, user_feedback, selected_model):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = 'src/Data/feedback_results.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp'] + list(data.keys()) + 
                            ['XGBoost_pred', 'LogisticRegression_pred', 'StackModel_pred', 'Otros_pred', 'feedback_satisfaction', 'selected_model'])
        writer.writerow([timestamp] + list(data.values()) + [
            predictions['XGBoost'],
            predictions['Logistic Regression'],
            predictions['Stack Model'],
            predictions['Red Neuronal'],
            1 if user_feedback == 'Satisfecho' else 0,
            selected_model  # Agregar el modelo seleccionado al CSV
        ])

# Nueva función para mostrar predicción y botón de guardado
def display_and_save_results(model_name, prediction, model_key):
    st.header(model_name)
    if prediction is not None:
        st.write(f"Predicción: {'Satisfecho' if prediction == 1 else 'Insatisfecho'}")
    else:
        st.write("Predicción no disponible")
    
    # Botón para guardar los resultados
    if st.button(f'Guardar resultados {model_name}'):
        st.session_state.selected_model = model_key
        save_to_csv(st.session_state.data, st.session_state.predictions, satisfaction, st.session_state.selected_model)
        st.success(f'Resultados guardados con el modelo {model_name}')

# Crear la aplicación Streamlit
st.title('Encuesta de Satisfacción')


# Crear el formulario
st.header('Por favor, rellene el siguiente formulario')

# Dividir el formulario en columnas
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

# Inputs de selección en dos columnas
with col1:
    gender = st.selectbox('Género', ['Masculino', 'Femenino'])
    customer_type = st.selectbox('Tipo de Cliente', ['Leal', 'Desleal'])
    age = st.number_input('Edad', min_value=0, max_value=120)
with col2:
    type_of_travel = st.selectbox('Tipo de Viaje', ['Personal', 'Negocios'])
    class_type = st.selectbox('Clase', ['Económica', 'Económica Plus', 'Negocios'])
    flight_distance = st.number_input('Distancia de Vuelo', min_value=0)

# Sliders en dos columnas
with col3:
    inflight_wifi_service = st.slider('Servicio de WiFi a bordo', 0, 5)
    departure_arrival_time_convenient = st.slider('Conveniencia del horario de salida/llegada', 0, 5)
    ease_of_online_booking = st.slider('Facilidad de reserva en línea', 0, 5)
    gate_location = st.slider('Ubicación de la puerta', 0, 5)
    food_and_drink = st.slider('Comida y bebida', 0, 5)
    online_boarding = st.slider('Embarque en línea', 0, 5)
    seat_comfort = st.slider('Confort del asiento', 0, 5)
    
with col4:
    inflight_entertainment = st.slider('Entretenimiento a bordo', 0, 5)
    onboard_service = st.slider('Servicio a bordo', 0, 5)
    leg_room_service = st.slider('Espacio para las piernas', 0, 5)
    baggage_handling = st.slider('Manejo de equipaje', 0, 5)
    checkin_service = st.slider('Servicio de check-in', 0, 5)
    inflight_service = st.slider('Servicio durante el vuelo', 0, 5)
    cleanliness = st.slider('Limpieza', 0, 5)

# Inputs numéricos en dos columnas
with col5:
    departure_delay = st.number_input('Retraso en la salida (minutos)', min_value=0)

with col6:
    arrival_delay = st.number_input('Retraso en la llegada (minutos)', min_value=0)

# Nueva pregunta sobre la satisfacción general con la aerolínea (ahora como select box)
satisfaction = st.selectbox(
    "En general, ¿cómo se encuentra con el servicio de la aerolínea?",
    ('Satisfecho', 'Insatisfecho')
)

# Inicializar variables de estado
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# Cuando el usuario selecciona su satisfacción, hacer las predicciones
if satisfaction in ['Satisfecho', 'Insatisfecho']:
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

    # Hacer las predicciones
    st.session_state.predictions = {
        'XGBoost': predict(data_df, xgboost_model),
        'Logistic Regression': predict(data_df, logistic_model),
        'Stack Model': predict(data_df, stack_model),
        'Red Neuronal': predict(data_df, neuronal_model)
    }
    st.session_state.data = data

    # Crear pestañas
    tab1, tab2, tab3, tab4 = st.tabs(["XGBoost", "Logistic Regression", "Stack Model", "Red Neuronal"])

    # Mostrar resultados y botón de guardado para cada pestaña usando la nueva función
    with tab1:
        display_and_save_results("XGBoost", st.session_state.predictions['XGBoost'], "XGBoost")

    with tab2:
        display_and_save_results("Logistic Regression", st.session_state.predictions['Logistic Regression'], "Logistic Regression")

    with tab3:
        display_and_save_results("Stack Model", st.session_state.predictions['Stack Model'], "Stack Model")

    with tab4:
        display_and_save_results("Red Neuronal", st.session_state.predictions['Red Neuronal'], "Red Neuronal")
