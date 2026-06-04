from pydantic import BaseModel

class CustomerData(BaseModel):
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    StdTransactionAmount: float

class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int

    