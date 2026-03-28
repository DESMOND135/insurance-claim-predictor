import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = FastAPI(title="Insurance Claim Predictor")

class InsuranceInput(BaseModel):
    age: int
    sex: str
    weight: int
    bmi: float
    hereditary_diseases: str
    no_of_dependents: int
    smoker: int
    city: str
    bloodpressure: int
    diabetes: int
    regular_ex: int
    job_title: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Claim Prediction API. Go to /docs for testing."}

@app.post("/predict")
def predict_claim(data: InsuranceInput):
    try:
        custom_data = CustomData(
            age=data.age,
            sex=data.sex,
            weight=data.weight,
            bmi=data.bmi,
            hereditary_diseases=data.hereditary_diseases,
            no_of_dependents=data.no_of_dependents,
            smoker=data.smoker,
            city=data.city,
            bloodpressure=data.bloodpressure,
            diabetes=data.diabetes,
            regular_ex=data.regular_ex,
            job_title=data.job_title
        )
        
        pred_df = custom_data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        
        return {"predicted_claim": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
