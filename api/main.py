from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("../models/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="ML Inference API")

# Define input data schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float  # Change according to your dataset

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "API is running"}

# Predict endpoint
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
