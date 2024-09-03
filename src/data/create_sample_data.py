import pandas as pd
import numpy as np
import os

# Crear un DataFrame de muestra
np.random.seed(0)
data = {
    'Age': np.random.randint(18, 80, 1000),
    'Flight Distance': np.random.randint(100, 5000, 1000),
    'Inflight wifi service': np.random.randint(0, 6, 1000),
    'Departure/Arrival time convenient': np.random.randint(0, 6, 1000),
    'Ease of Online booking': np.random.randint(0, 6, 1000),
    'Gate location': np.random.randint(0, 6, 1000),
    'Food and drink': np.random.randint(0, 6, 1000),
    'Online boarding': np.random.randint(0, 6, 1000),
    'Seat comfort': np.random.randint(0, 6, 1000),
    'Inflight entertainment': np.random.randint(0, 6, 1000),
    'On-board service': np.random.randint(0, 6, 1000),
    'Leg room service': np.random.randint(0, 6, 1000),
    'Baggage handling': np.random.randint(0, 6, 1000),
    'Checkin service': np.random.randint(0, 6, 1000),
    'Inflight service': np.random.randint(0, 6, 1000),
    'Cleanliness': np.random.randint(0, 6, 1000),
    'Departure Delay in Minutes': np.random.randint(0, 300, 1000),
    'Arrival Delay in Minutes': np.random.randint(0, 300, 1000),
    'satisfaction': np.random.choice(['satisfied', 'neutral or dissatisfied'], 1000)
}

df = pd.DataFrame(data)

# Guardar el DataFrame como CSV
output_path = os.path.join('data', 'raw', 'airline_satisfaction.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Archivo de muestra creado en: {output_path}")
print(f"Distribución de 'satisfaction':\n{df['satisfaction'].value_counts(normalize=True)}")