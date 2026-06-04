from fastapi import FastAPI
from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse
)
import joblib

app = FastAPI(
    title="Credit Risk API",
    version="1.0"
)

model = joblib.load("models/best_model.pkl")

@app.get("/")
def home():
    return {"message": "Credit Risk API Running"}

@app.post("/predict",
          response_model=PredictionResponse)
def predict(data: CustomerData):

    features = [[
        data.TotalTransactionAmount,
        data.AverageTransactionAmount,
        data.TransactionCount,
        data.StdTransactionAmount
    ]]

    probability = model.predict_proba(features)[0][1]
    prediction = model.predict(features)[0]

    return {
        "risk_probability": float(probability),
        "prediction": int(prediction)
    }