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
def save_to_csv(data, predictions, user_agreement):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = 'src/Data/feedback_results.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp'] + list(data.keys()) + 
                            ['XGBoost_pred', 'LogisticRegression_pred', 'StackModel_pred', 'RedNeuronal_pred', 'feedback_satisfaction'])
        writer.writerow([timestamp] + list(data.values()) + [
            predictions['Modelo1'],
            predictions['Modelo2'],
            predictions['Modelo3'],
            predictions['Modelo4'],
            1 if user_agreement == 'Sí' else 0
        ])

# Crear la aplicación Streamlit
st.title('Encuesta de Satisfacción')

# Bienvenida personalizada
st.markdown("""
**¡Gracias por volar con nosotros!** 
Te invitamos a compartir tu experiencia en esta breve encuesta (menos de 2 minutos). Tu opinión es muy valiosa para mejorar nuestros servicios.
""")

# Crear el formulario
#st.header('Por favor, rellene el siguiente formulario')

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

def satisfaction_radio(label, var_name):
    emojis = ['😡', '😠', '😞', '😐', '😊', '😁']  # Emojis de 0 a 5
    st.write(label)  # Mostrar el título

    # Crear radio buttons en fila horizontal
    selected_value = st.radio(
        label,
        options=emojis,
        index=0,  # Establecer valor inicial
        key=var_name,
        horizontal=True  # Opciones en horizontal
    )
    return emojis.index(selected_value)

# Columna 3: Botones para satisfacción
inflight_wifi_service = satisfaction_radio('Servicio de WiFi a bordo', 'inflight_wifi_service')
departure_arrival_time_convenient = satisfaction_radio('Conveniencia del horario de salida/llegada', 'departure_arrival_time_convenient')
ease_of_online_booking = satisfaction_radio('Facilidad de reserva en línea', 'ease_of_online_booking')
gate_location = satisfaction_radio('Ubicación de la puerta', 'gate_location')
food_and_drink = satisfaction_radio('Comida y bebida', 'food_and_drink')
online_boarding = satisfaction_radio('Embarque en línea', 'online_boarding')
seat_comfort = satisfaction_radio('Confort del asiento', 'seat_comfort')

# Columna 4: Radio buttons para satisfacción
inflight_entertainment = satisfaction_radio('Entretenimiento a bordo', 'inflight_entertainment')
onboard_service = satisfaction_radio('Servicio a bordo', 'onboard_service')
leg_room_service = satisfaction_radio('Espacio para las piernas', 'leg_room_service')
baggage_handling = satisfaction_radio('Manejo de equipaje', 'baggage_handling')
checkin_service = satisfaction_radio('Servicio de check-in', 'checkin_service')
inflight_service = satisfaction_radio('Servicio durante el vuelo', 'inflight_service')
cleanliness = satisfaction_radio('Limpieza', 'cleanliness')
# Inputs numéricos en dos columnas
with col5:
    departure_delay = st.number_input('Retraso en la salida (minutos)', min_value=0)

with col6:
    arrival_delay = st.number_input('Retraso en la llegada (minutos)', min_value=0)

# Inicializar variables de estado
if 'results_shown' not in st.session_state:
    st.session_state.results_shown = False
if 'user_agreement' not in st.session_state:
    st.session_state.user_agreement = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Botón para hacer la predicción
if st.button('Ver Resultados'):
    # Preparar los datos para la predicción
    st.session_state.data = {
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
    data_df = pd.DataFrame([st.session_state.data])

    # Hacer las predicciones
    st.session_state.predictions = {
        'Modelo1': predict(data_df, xgboost_model),
        'Modelo2': predict(data_df, logistic_model),
        'Modelo3': predict(data_df, stack_model),
        'Modelo4': predict(data_df, neuronal_model)
    }

    st.session_state.results_shown = True

# Mostrar resultados si están disponibles
if st.session_state.results_shown:
    # Crear una tabla con los resultados
    results_df = pd.DataFrame({
        'Predicción': [('Satisfecho' if pred == 1 else 'Insatisfecho') for pred in st.session_state.predictions.values()]
    }, index=st.session_state.predictions.keys())
    
    st.subheader("Resultados de las predicciones")
    st.table(results_df)

    # Preguntar al usuario si está de acuerdo con las predicciones
    st.session_state.user_agreement = st.radio("¿Está de acuerdo con estas predicciones, las cuales se realizaron utilizando un modelo de machine learning?", ('Sí', 'No'))

    # Mostrar el botón "Enviar Formulario" solo si el usuario ha seleccionado una opción
    if st.session_state.user_agreement:
        if st.button('Enviar Formulario'):
            save_to_csv(st.session_state.data, st.session_state.predictions, st.session_state.user_agreement)
            #st.success('Formulario enviado y resultados guardados exitosamente.')
            # Reiniciar las variables de estado
            st.session_state.results_shown = False
            st.session_state.user_agreement = None
            st.session_state.data = None
            st.session_state.predictions = None
            

            # Mensaje de agradecimiento
            st.success('¡Muchas gracias por tu colaboración! Tu opinión nos ayudará a mejorar tu próxima experiencia de vuelo.')
