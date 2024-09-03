import pandas as pd
import os

def analyze_data(filepath):
    print(f"Analizando datos en: {filepath}")
    
    # Leer solo las primeras filas para obtener los nombres de las columnas
    df_head = pd.read_csv(filepath, nrows=5)
    print("\nPrimeras 5 filas:")
    print(df_head)
    
    # Leer todo el archivo
    df = pd.read_csv(filepath)
    
    print(f"\nNúmero total de filas: {len(df)}")
    print(f"\nColumnas en el dataset: {', '.join(df.columns)}")
    
    if 'satisfaction' in df.columns:
        print("\nDistribución de la columna 'satisfaction':")
        print(df['satisfaction'].value_counts(normalize=True))
    else:
        print("\nLa columna 'satisfaction' no está presente en el dataset.")
    
    print("\nResumen estadístico de las columnas numéricas:")
    print(df.describe())

if __name__ == "__main__":
    # Asegúrate de que esta ruta sea correcta
    data_path = os.path.join('data', 'processed', 'clean_airline_satisfaction.csv')
    analyze_data(data_path)