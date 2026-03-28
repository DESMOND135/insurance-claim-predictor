import os
import sys

# Ensure src module is imported correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def test_model_exists():
    assert os.path.exists(os.path.join("artifacts", "model.pkl")), "Model file missing"
    assert os.path.exists(os.path.join("artifacts", "preprocessor.pkl")), "Preprocessor missing"

def test_pipeline_prediction():
    pipeline = PredictPipeline()
    custom_data = CustomData(
        age=30, sex="male", weight=70, bmi=25.0,
        hereditary_diseases="NoDisease", no_of_dependents=0,
        smoker=0, city="NewYork", bloodpressure=72,
        diabetes=0, regular_ex=1, job_title="Student"
    )
    df = custom_data.get_data_as_data_frame()
    preds = pipeline.predict(df)
    
    assert preds is not None
    assert len(preds) == 1
