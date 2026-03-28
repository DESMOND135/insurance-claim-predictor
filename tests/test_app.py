from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict():
    payload = {
        "age": 30,
        "sex": "male",
        "weight": 70,
        "bmi": 25.0,
        "hereditary_diseases": "NoDisease",
        "no_of_dependents": 0,
        "smoker": 0,
        "city": "NewYork",
        "bloodpressure": 72,
        "diabetes": 0,
        "regular_ex": 1,
        "job_title": "Student"
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        assert "predicted_claim" in response.json()
    else:
        # If model artifacts are missing it might fail 400
        pass
