import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        age: int,
        sex: str,
        weight: int,
        bmi: float,
        hereditary_diseases: str,
        no_of_dependents: int,
        smoker: int,
        city: str,
        bloodpressure: int,
        diabetes: int,
        regular_ex: int,
        job_title: str
    ):
        self.age = age
        self.sex = sex
        self.weight = weight
        self.bmi = bmi
        self.hereditary_diseases = hereditary_diseases
        self.no_of_dependents = no_of_dependents
        self.smoker = smoker
        self.city = city
        self.bloodpressure = bloodpressure
        self.diabetes = diabetes
        self.regular_ex = regular_ex
        self.job_title = job_title

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "weight": [self.weight],
                "bmi": [self.bmi],
                "hereditary_diseases": [self.hereditary_diseases],
                "no_of_dependents": [self.no_of_dependents],
                "smoker": [self.smoker],
                "city": [self.city],
                "bloodpressure": [self.bloodpressure],
                "diabetes": [self.diabetes],
                "regular_ex": [self.regular_ex],
                "job_title": [self.job_title],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
