import joblib
import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_interaction_features(X):
    """
    Crea características de interacción.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    if 'Inflight wifi service' in X.columns and 'Inflight entertainment' in X.columns:
        X['wifi_entertainment'] = X['Inflight wifi service'] * X['Inflight entertainment']
    
    service_columns = ['Food and drink', 'Inflight service', 'On-board service']
    if all(col in X.columns for col in service_columns):
        X['total_service'] = X[service_columns].sum(axis=1)
    
    return X

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo del modelo no se encuentra en {model_path}.")
    logging.info(f"Cargando modelo desde {model_path}")
    return joblib.load(model_path)

def prepare_input(input_data):
    expected_columns = ['Age', 'Flight Distance', 'Inflight wifi service',
                        'Departure/Arrival time convenient', 'Ease of Online booking',
                        'Gate location', 'Food and drink', 'Online boarding',
                        'Seat comfort', 'Inflight entertainment', 'On-board service',
                        'Leg room service', 'Baggage handling', 'Checkin service',
                        'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
                        'Arrival Delay in Minutes']
    
    df = pd.DataFrame([input_data])
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_columns]
    
    # Crear características adicionales
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    df['service_rating'] = df[['Inflight wifi service', 'Food and drink', 'Inflight entertainment', 'On-board service']].mean(axis=1)
    df['convenience_rating'] = df[['Online boarding', 'Gate location', 'Ease of Online booking']].mean(axis=1)
    
    # Aplicar create_interaction_features
    df = create_interaction_features(df)
    
    # Filtrar las columnas para que coincidan con las del modelo
    model_features = [
        'Age', 'Flight Distance', 'Inflight wifi service',
        'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding',
        'Seat comfort', 'Inflight entertainment', 'On-board service',
        'Leg room service', 'Baggage handling', 'Checkin service',
        'Inflight service', 'Cleanliness', 'Departure Delay in Minutes',
        'Arrival Delay in Minutes', 'total_delay', 'service_rating', 'convenience_rating'
    ]
    
    df = df[model_features]
    
    logging.info(f"Datos de entrada preparados. Shape: {df.shape}")
    return df

def predict(model, input_data):
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    logging.info(f"Predicción realizada. Resultado: {prediction}")
    return prediction, probabilities

def main():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(project_dir, 'models', 'best_model.joblib')
    
    try:
        model = load_model(model_path)
        
        # Ejemplo de datos de entrada
        input_data = {
            'Age': 30,
            'Flight Distance': 1000,
            'Inflight wifi service': 3,
            'Departure/Arrival time convenient': 4,
            'Ease of Online booking': 3,
            'Gate location': 3,
            'Food and drink': 4,
            'Online boarding': 3,
            'Seat comfort': 4,
            'Inflight entertainment': 3,
            'On-board service': 4,
            'Leg room service': 3,
            'Baggage handling': 4,
            'Checkin service': 4,
            'Inflight service': 4,
            'Cleanliness': 4,
            'Departure Delay in Minutes': 15,
            'Arrival Delay in Minutes': 20
        }
        
        prepared_input = prepare_input(input_data)
        prediction, probabilities = predict(model, prepared_input)
        
        print(f"Predicción: {'Satisfecho' if prediction[0] == 1 else 'Insatisfecho'}")
        print(f"Probabilidad de estar satisfecho: {probabilities[0][1]:.2f}")
        print(f"Probabilidad de estar insatisfecho: {probabilities[0][0]:.2f}")
        
    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        logging.error(f"Directorio actual: {os.getcwd()}")
        logging.error("Asegúrate de haber ejecutado el script de entrenamiento (train_model.py) "
                      "y de estar ejecutando este script desde el directorio raíz del proyecto.")
    except Exception as e:
        logging.error(f"Ocurrió un error inesperado: {e}")
        logging.error("Detalles del error:", exc_info=True)

if __name__ == "__main__":
    main()