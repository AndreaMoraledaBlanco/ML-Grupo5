import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def plot_feature_distributions(df):
    # Crear el directorio si no existe
    os.makedirs('reports/figures', exist_ok=True)
    
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        if df[column].dtype == 'object':
            sns.countplot(x=column, data=df)
        else:
            sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        # Reemplazar caracteres problemáticos en el nombre del archivo
        safe_column_name = "".join([c if c.isalnum() else "_" for c in column])
        plt.savefig(f'reports/figures/{safe_column_name}_distribution.png')
        plt.close()

def plot_correlation_matrix(df):
    # Crear el directorio si no existe
    os.makedirs('reports/figures', exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('reports/figures/correlation_matrix.png')
    plt.close()

def main():
    # Asegúrate de que esta ruta sea correcta
    df = load_data('data/processed/featured_airline_satisfaction.csv')
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    print("EDA completed. Check the 'reports/figures' directory for visualizations.")

if __name__ == "__main__":
    main()