import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

def predict_risk(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]

    return {
        "risk_prediction": int(prediction[0]),
        "risk_probability": float(probability[0])
    }
