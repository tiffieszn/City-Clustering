from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.model_utils import predict_cluster

class CityPayload(BaseModel):
    population_density: float
    avg_income: float
    internet_penetration: float
    avg_rent: float
    air_quality_index: float
    public_transport_score: float
    happiness_score: float
    green_space_ratio: float

app = FastAPI(title='Smart City Clustering API')

@app.get('/')
def root():
    return {'message':'Smart City Clustering API - send POST to /predict'}

@app.post('/predict')
def predict(payload: CityPayload):
    data = payload.dict()
    res = predict_cluster(data)
    return res