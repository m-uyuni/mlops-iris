from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Load model
model = joblib.load("models/model.joblib")

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

@app.post("/predict")
def predict(features: IrisFeatures):
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = int(model.predict(data)[0])
    return {"prediction": prediction}