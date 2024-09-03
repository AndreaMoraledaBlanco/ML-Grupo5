import pandas as pd
import numpy as np

def create_interaction_features(df):
    df['wifi_entertainment'] = df['Inflight wifi service'] * df['Inflight entertainment']
    df['total_service'] = df[['Food and drink', 'Inflight service', 'On-board service']].sum(axis=1)
    return df

def load_data(filepath):
    return pd.read_csv(filepath)

def create_total_delay(df):
    df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
    return df

def create_service_rating(df):
    service_columns = ['Inflight wifi service', 'Food and drink', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness']
    df['average_service_rating'] = df[service_columns].mean(axis=1)
    return df

def main():
    input_filepath = 'data/processed/clean_airline_satisfaction.csv'
    output_filepath = 'data/processed/featured_airline_satisfaction.csv'
    
    df = load_data(input_filepath)
    df = create_total_delay(df)
    df = create_service_rating(df)
    
    df.to_csv(output_filepath, index=False)
    print(f"Featured data saved to {output_filepath}")

if __name__ == "__main__":
    main()