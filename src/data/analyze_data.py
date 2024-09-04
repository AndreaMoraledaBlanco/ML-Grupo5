import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_data(filepath):
    print(f"Analizando datos en: {filepath}")
    
    df = pd.read_csv(filepath)
    
    print(f"\nNúmero total de filas: {len(df)}")
    print(f"\nColumnas en el dataset: {', '.join(df.columns)}")
    
    print("\nInformación del DataFrame:")
    print(df.info())
    
    print("\nResumen estadístico de las columnas numéricas:")
    print(df.describe())
    
    if 'satisfaction' in df.columns:
        print("\nDistribución de la columna 'satisfaction':")
        print(df['satisfaction'].value_counts(normalize=True))
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x='satisfaction', data=df)
        plt.title('Distribución de Satisfacción')
        plt.savefig('reports/figures/satisfaction_distribution.png')
        plt.close()
    else:
        print("\nLa columna 'satisfaction' no está presente en el dataset.")
    
    print("\nCorrelación entre variables numéricas:")
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.savefig('reports/figures/correlation_matrix.png')
    plt.close()
    
    print("\nGráficos de distribución para variables numéricas guardados en 'reports/figures/'")
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribución de {col}')
        plt.savefig(f'reports/figures/{col}_distribution.png')
        plt.close()

if __name__ == "__main__":
    data_path = os.path.join('data', 'processed', 'clean_airline_satisfaction.csv')
    analyze_data(data_path)