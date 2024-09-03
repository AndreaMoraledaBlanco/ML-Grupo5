import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import os
import logging

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    logging.info(f"Cargando datos desde {filepath}")
    
    if not os.path.exists(filepath):
        logging.error(f"Error: El archivo {filepath} no existe.")
        return None
    
    try:
        df = pd.read_csv(filepath, on_bad_lines='warn')
        if df.empty:
            logging.error(f"Error: El archivo {filepath} está vacío o no contiene datos válidos.")
            return None
        
        logging.info(f"Forma del DataFrame: {df.shape}")
        logging.info(f"Columnas: {df.columns.tolist()}")
        logging.info(f"Primeras filas:\n{df.head()}")
        
        return df
    except pd.errors.EmptyDataError:
        logging.error(f"Error: El archivo {filepath} está vacío o no contiene datos válidos.")
        return None
    except Exception as e:
        logging.error(f"Error al cargar los datos: {e}")
        return None

def handle_missing_values(df):
    logging.info("Manejando valores faltantes...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=[object]).columns

    # Usar IterativeImputer para variables numéricas
    imputer = IterativeImputer(random_state=42)
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    return df

def handle_outliers(df, columns, method='iqr'):
    logging.info("Manejando valores atípicos...")
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def create_new_features(df):
    logging.info("Creando nuevas características...")
    if 'Departure Delay in Minutes' in df.columns and 'Arrival Delay in Minutes' in df.columns:
        df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    
    service_columns = ['Inflight wifi service', 'Food and drink', 'Inflight entertainment', 'On-board service']
    if all(col in df.columns for col in service_columns):
        df['service_rating'] = df[service_columns].mean(axis=1)
    
    convenience_columns = ['Online boarding', 'Gate location', 'Ease of Online booking']
    if all(col in df.columns for col in convenience_columns):
        df['convenience_rating'] = df[convenience_columns].mean(axis=1)
    
    if 'Customer Type' in df.columns:
        df['is_loyal_customer'] = np.where(df['Customer Type'] == 'Loyal Customer', 1, 0)
    else:
        logging.warning("La columna 'Customer Type' no está presente. No se pudo crear 'is_loyal_customer'.")
    
    return df

def encode_categorical_variables(df):
    logging.info("Codificando variables categóricas...")
    categorical_columns = df.select_dtypes(include=[object]).columns
    
    for column in categorical_columns:
        if column != 'satisfaction':
            df = pd.get_dummies(df, columns=[column], prefix=column, drop_first=True)
    
    return df

def scale_numeric_features(df):
    logging.info("Escalando características numéricas...")
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col != 'satisfaction']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def transform_skewed_features(df):
    logging.info("Transformando características asimétricas...")
    skewed_features = ['Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Flight Distance']
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    df[skewed_features] = pt.fit_transform(df[skewed_features])
    return df

def balance_classes(X, y):
    logging.info("Balanceando clases...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def preprocess_data(input_filepath, output_filepath):
    logging.info(f"Cargando datos desde {input_filepath}")
    df = load_data(input_filepath)
    
    if df is None:
        return None
    
     # Preprocesamiento de datos
    df = handle_missing_values(df)
    df = handle_outliers(df, [col for col in ['Departure Delay in Minutes', 'Arrival Delay in Minutes'] if col in df.columns])
    df = create_new_features(df)
    df = encode_categorical_variables(df)
    
    # Codificar la variable 'satisfaction'
    if 'satisfaction' in df.columns:
        df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0}).astype(int)
    else:
        logging.error("La columna 'satisfaction' no está presente en el dataset.")
        return None
    
    df = transform_skewed_features(df)
    df = scale_numeric_features(df)
    
    # Separación de características y variable objetivo
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    
    # Balanceo de clases
    X_resampled, y_resampled = balance_classes(X, y)
    
    # Combinar características y variable objetivo balanceadas
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    
    logging.info(f"Guardando datos preprocesados en {output_filepath}")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df_resampled.to_csv(output_filepath, index=False)
    logging.info("Preprocesamiento completado.")
    logging.info(f"Distribución de 'satisfaction' después del preprocesamiento:\n{df_resampled['satisfaction'].value_counts(normalize=True)}")
    logging.info(f"Tipos de datos después del preprocesamiento:\n{df_resampled.dtypes}")

    return df_resampled

if __name__ == "__main__":
    current_dir = os.getcwd()
    logging.info(f"Directorio de trabajo actual: {current_dir}")

    input_filepath = os.path.abspath("data/raw/airline_satisfaction.csv")
    output_filepath = os.path.abspath("data/processed/clean_airline_satisfaction.csv")
    
    logging.info(f"Ruta absoluta del archivo de entrada: {input_filepath}")
    logging.info(f"Ruta absoluta del archivo de salida: {output_filepath}")

    if os.path.exists(input_filepath):
        logging.info(f"El archivo de entrada existe. Tamaño: {os.path.getsize(input_filepath)} bytes")
    else:
        logging.error(f"El archivo de entrada {input_filepath} no existe.")

    preprocess_data(input_filepath, output_filepath)