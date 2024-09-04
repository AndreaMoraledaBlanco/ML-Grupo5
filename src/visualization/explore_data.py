import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def plot_feature_distributions(df):
    os.makedirs('reports/figures', exist_ok=True)
    
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        if df[column].dtype == 'object':
            sns.countplot(x=column, data=df)
            plt.xticks(rotation=45)
        else:
            sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        safe_column_name = "".join([c if c.isalnum() else "_" for c in column])
        plt.tight_layout()
        plt.savefig(f'reports/figures/{safe_column_name}_distribution.png')
        plt.close()

def plot_correlation_matrix(df):
    os.makedirs('reports/figures', exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png')
    plt.close()

def plot_pairplot(df):
    os.makedirs('reports/figures', exist_ok=True)
    
    sns.pairplot(df, hue='satisfaction', vars=['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'])
    plt.tight_layout()
    plt.savefig('reports/figures/pairplot.png')
    plt.close()

def plot_boxplots(df):
    os.makedirs('reports/figures', exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='satisfaction', y=col, data=df)
        plt.title(f'Boxplot of {col} by Satisfaction')
        plt.tight_layout()
        plt.savefig(f'reports/figures/boxplot_{col}.png')
        plt.close()

def main():
    df = load_data('data/processed/featured_airline_satisfaction.csv')
    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    plot_pairplot(df)
    plot_boxplots(df)
    print("EDA completed. Check the 'reports/figures' directory for visualizations.")

if __name__ == "__main__":
    main()