from fastapi import FastAPI
from pydantic import BaseModel
from src.Modelos.logistic_model import LogisticModel
from src.Modelos.xgboost_model import XGBoostModel
from src.database.connection import create_connection, close_connection
import pandas as pd

app = FastAPI()

# Cargar modelos
logistic_model = LogisticModel.load_model('src/Modelos/logistic_model.joblib')
xgboost_model = XGBoostModel.load_model('src/Modelos/xgboost_model.joblib')

class PredictionInput(BaseModel):
    gender: str
    customer_type: str
    age: int
    travel_type: str
    flight_distance: float
    inflight_wifi: int
    departure_convenience: int
    online_booking: int
    gate_location: int
    food_drink: int
    online_boarding: int
    seat_comfort: int
    inflight_entertainment: int
    onboard_service: int
    legroom_service: int
    baggage_handling: int
    checkin_service: int
    inflight_service: int
    cleanliness: int
    departure_delay: int
    arrival_delay: int

class FeedbackInput(BaseModel):
    rating: int
    comments: str

@app.post("/predict")
def predict(data: PredictionInput):
    inputs = pd.DataFrame([data.dict()])
    
    logistic_pred, logistic_prob = predict_satisfaction(logistic_model, inputs)
    xgboost_pred, xgboost_prob = predict_satisfaction(xgboost_model, inputs)
    
    save_prediction(inputs, "Logistic", logistic_pred, logistic_prob)
    save_prediction(inputs, "XGBoost", xgboost_pred, xgboost_prob)
    
    return {
        "logistic_prediction": int(logistic_pred),
        "xgboost_prediction": int(xgboost_pred)
    }

@app.post("/feedback")
def feedback(data: FeedbackInput):
    save_feedback(data.rating, data.comments)
    return {"message": "Feedback received and saved"}

# Funciones auxiliares (predict_satisfaction, save_prediction, save_feedback) 
def predict_satisfaction(model, inputs):
    proba = model.predict_proba(inputs)[0]
    prediction = 1 if proba[1] > 0.5 else 0
    return prediction, proba[1]

def save_prediction(inputs, model_name, prediction, probability):
    connection = create_connection()
    cursor = connection.cursor()
    
    query = """
    INSERT INTO predictions (model, prediction, probability, gender, customer_type, age, travel_type, flight_distance, 
    inflight_wifi, departure_convenience, online_booking, gate_location, food_drink, online_boarding, seat_comfort, 
    inflight_entertainment, onboard_service, legroom_service, baggage_handling, checkin_service, inflight_service_personal, 
    cleanliness, departure_delay, arrival_delay)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    values = (
        model_name, prediction, probability,
        inputs['Gender'].values[0], inputs['Customer Type'].values[0], inputs['Age'].values[0],
        inputs['Type of Travel'].values[0], inputs['Flight Distance'].values[0], inputs['Inflight wifi service'].values[0],
        inputs['Departure/Arrival time convenient'].values[0], inputs['Ease of Online booking'].values[0],
        inputs['Gate location'].values[0], inputs['Food and drink'].values[0], inputs['Online boarding'].values[0],
        inputs['Seat comfort'].values[0], inputs['Inflight entertainment'].values[0], inputs['On-board service'].values[0],
        inputs['Leg room service'].values[0], inputs['Baggage handling'].values[0], inputs['Checkin service'].values[0],
        inputs['Inflight service'].values[0], inputs['Cleanliness'].values[0], inputs['Departure Delay in Minutes'].values[0],
        inputs['Arrival Delay in Minutes'].values[0]
    )
    
    cursor.execute(query, values)
    connection.commit()
    close_connection(connection)

def save_feedback(rating, comment):
    connection = create_connection()
    cursor = connection.cursor()
    
    query = "INSERT INTO feedback (rating, comments) VALUES (%s, %s)"
    values = (rating, comment)
    
    cursor.execute(query, values)
    connection.commit()
    close_connection(connection)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)