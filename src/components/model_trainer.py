import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_path):
        try:
            logging.info("Starting data ingestion and preprocessing")
            df = pd.read_csv(train_path)

            logging.info("Read dataset. Shape: %s", df.shape)

            target_column_name = "claim"
            X = df.drop(columns=[target_column_name], axis=1)
            y = df[target_column_name]

            # Categorical vs Numerical features
            num_features = X.select_dtypes(exclude="object").columns
            cat_features = X.select_dtypes(include="object").columns
            logging.info(f"Numerical features: {num_features}")
            logging.info(f"Categorical features: {cat_features}")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info("Applying preprocessing object on training and testing datasets")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.model_trainer_config.preprocessor_file_path,
                obj=preprocessor
            )

            logging.info("Training model...")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_arr, y_train)

            logging.info("Model Training completed. Evaluating...")
            y_train_pred = model.predict(X_train_arr)
            y_test_pred = model.predict(X_test_arr)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            logging.info(f"Train R2: {train_r2}")
            logging.info(f"Test R2: {test_r2}")

            logging.info("Saving trained model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return train_r2, test_r2

        except Exception as e:
            raise CustomException(e, sys)
