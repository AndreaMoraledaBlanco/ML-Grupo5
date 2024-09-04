import pandas as pd
import numpy as np
import os

np.random.seed(42)

def create_sample_data(n_samples=1000):
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Flight Distance': np.random.randint(100, 5000, n_samples),
        'Inflight wifi service': np.random.randint(0, 6, n_samples),
        'Departure/Arrival time convenient': np.random.randint(0, 6, n_samples),
        'Ease of Online booking': np.random.randint(0, 6, n_samples),
        'Gate location': np.random.randint(0, 6, n_samples),
        'Food and drink': np.random.randint(0, 6, n_samples),
        'Online boarding': np.random.randint(0, 6, n_samples),
        'Seat comfort': np.random.randint(0, 6, n_samples),
        'Inflight entertainment': np.random.randint(0, 6, n_samples),
        'On-board service': np.random.randint(0, 6, n_samples),
        'Leg room service': np.random.randint(0, 6, n_samples),
        'Baggage handling': np.random.randint(0, 6, n_samples),
        'Checkin service': np.random.randint(0, 6, n_samples),
        'Inflight service': np.random.randint(0, 6, n_samples),
        'Cleanliness': np.random.randint(0, 6, n_samples),
        'Departure Delay in Minutes': np.random.exponential(scale=15, size=n_samples).astype(int),
        'Arrival Delay in Minutes': np.random.exponential(scale=15, size=n_samples).astype(int),
    }
    
    df = pd.DataFrame(data)
    
    # Crear 'satisfaction' basado en otras variables para simular una relación
    df['satisfaction'] = ((df['Seat comfort'] + df['Inflight service'] + df['Cleanliness']) / 3 > 3).astype(int)
    df['satisfaction'] = df['satisfaction'].map({0: 'neutral or dissatisfied', 1: 'satisfied'})
    
    return df

def main():
    df = create_sample_data()
    output_path = os.path.join('data', 'raw', 'airline_satisfaction.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Archivo de muestra creado en: {output_path}")
    print(f"Distribución de 'satisfaction':\n{df['satisfaction'].value_counts(normalize=True)}")

if __name__ == "__main__":
    main()