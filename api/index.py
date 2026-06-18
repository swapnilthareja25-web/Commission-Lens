from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="CommissionLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Load assets
try:
    model = joblib.load(os.path.join(DATA_DIR, 'xgboost_commission_model.pkl'))
    features_to_use = joblib.load(os.path.join(DATA_DIR, 'model_features.pkl'))
    fund_houses_list = joblib.load(os.path.join(DATA_DIR, 'fund_houses.pkl'))
except Exception as e:
    print(f"Error loading assets: {e}")
    model = None
    features_to_use = []
    fund_houses_list = []

class PredictionRequest(BaseModel):
    amc: str
    days_since_launch: int
    risk_90d: float
    return_7d: float
    return_30d: float
    return_90d: float

@app.get("/api/amcs")
def get_amcs():
    return {"amcs": fund_houses_list}

@app.post("/api/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    input_data = {
        'return_7d': request.return_7d,
        'return_30d': request.return_30d,
        'return_90d': request.return_90d,
        'days_since_launch': request.days_since_launch,
        'risk_90d': request.risk_90d
    }
    
    # Process AMC features
    for col in features_to_use:
        if 'fund_house_' in col:
            if col == f"fund_house_{request.amc}":
                input_data[col] = 1
            else:
                input_data[col] = 0
                
    # Create DataFrame with exact column order
    input_df = pd.DataFrame([input_data])[features_to_use]
    
    # Predict
    try:
        prediction = model.predict(input_df)[0]
        return {"predicted_drag": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
