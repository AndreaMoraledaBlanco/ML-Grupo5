import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
import json
from contextlib import contextmanager
from typing import Dict, Any

# Cargar las variables de entorno
load_dotenv()

# Configuración de la base de datos
DB_CONFIG = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME")
}

# Configuración de rutas y versión del modelo
DATA_CONFIG = {
    "customer_data": "C:\\Users\\Administrator\\Desktop\\IA_Bootcamp\\Proyecto_ML_Airlines\\src\\Data\\airline_recoded.csv",
    "predictions_data": "C:\\Users\\Administrator\\Desktop\\IA_Bootcamp\\Proyecto_ML_Airlines\\src\\Data\\XGBoost_predicciones.csv",
    "model_metrics": "C:\\Users\\Administrator\\Desktop\\IA_Bootcamp\\Proyecto_ML_Airlines\\src\\Data\\model_metrics.json",
    "model_version": "1.0"
}

@contextmanager
def get_db_connection(db_config: Dict[str, str]):
    """Context manager para manejar la conexión a la base de datos."""
    engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
    try:
        yield engine
    finally:
        engine.dispose()

def load_and_prepare_data(file_path: str, column_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """Carga y prepara los datos desde un archivo CSV."""
    df = pd.read_csv(file_path)
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)
    return df
def clean_table(table_name: str, engine: Any) -> None:
    """Limpia todos los datos de la tabla especificada."""
    with engine.connect() as connection:
        connection.execute(text(f"TRUNCATE TABLE {table_name}"))
    print(f"Tabla '{table_name}' limpiada correctamente.")

def insert_data_to_db(df: pd.DataFrame, table_name: str, engine: Any) -> None:
    """Limpia la tabla e inserta los nuevos datos."""
    clean_table(table_name, engine)
    df.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f"Datos insertados correctamente en la tabla '{table_name}'.")

def process_customer_data(engine: Any, file_path: str, column_mapping: Dict[str, str]) -> None:
    """Procesa e inserta los datos de clientes."""
    customer_data = load_and_prepare_data(file_path, column_mapping)
    insert_data_to_db(customer_data, 'customerdata', engine)

def process_predictions_data(engine: Any, file_path: str, model_version: str) -> None:
    """Procesa e inserta los datos de predicciones."""
    predictions_data = load_and_prepare_data(file_path)
    predictions_filtered = predictions_data[['Prediccion', 'Predicted_Probability']].copy()
    predictions_filtered['model_version'] = model_version
    predictions_filtered.rename(columns={
        'Prediccion': 'predicted_value',
        'Predicted_Probability': 'predicted_probability',
    }, inplace=True)
    insert_data_to_db(predictions_filtered, 'predictions', engine)

def process_model_performance(engine: Any, file_path: str, model_version: str) -> None:
    """Procesa e inserta los datos de rendimiento del modelo."""
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    metrics_df = pd.DataFrame({
        'model_version': [model_version] * 2,
        'metric_name': ['Accuracy', 'ROC AUC'],
        'metric_value': [metrics['accuracy'], metrics['roc_auc']]
    })
    insert_data_to_db(metrics_df, 'ModelPerformance', engine)

def main() -> None:
    customer_mapping = {
        'Gender': 'gender', 'Customer Type': 'customer_type', 'Age': 'age',
        'Type of Travel': 'type_of_travel', 'Class': 'class',
        'Flight Distance': 'flight_distance', 'Inflight wifi service': 'inflight_wifi_service',
        'Departure/Arrival time convenient': 'departure_arrival_time_convenient',
        'Ease of Online booking': 'ease_of_online_booking', 'Gate location': 'gate_location',
        'Food and drink': 'food_and_drink', 'Online boarding': 'online_boarding',
        'Seat comfort': 'seat_comfort', 'Inflight entertainment': 'inflight_entertainment',
        'On-board service': 'on_board_service', 'Leg room service': 'leg_room_service',
        'Baggage handling': 'baggage_handling', 'Checkin service': 'checkin_service',
        'Inflight service': 'inflight_service', 'Cleanliness': 'cleanliness',
        'Departure Delay in Minutes': 'departure_delay_in_minutes',
        'Arrival Delay in Minutes': 'arrival_delay_in_minutes',
        'Satisfaction': 'satisfaction'
    }

    with get_db_connection(DB_CONFIG) as engine:
        print(f"Conectado a la base de datos {DB_CONFIG['database']} en {DB_CONFIG['host']} como {DB_CONFIG['user']}")

        process_customer_data(engine, DATA_CONFIG['customer_data'], customer_mapping)
        process_predictions_data(engine, DATA_CONFIG['predictions_data'], DATA_CONFIG['model_version'])
        process_model_performance(engine, DATA_CONFIG['model_metrics'], DATA_CONFIG['model_version'])

if __name__ == "__main__":
    main()