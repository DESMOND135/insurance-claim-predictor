import os
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    csv_path = os.path.join(os.getcwd(), "insurance.csv")
    trainer = ModelTrainer()
    train_score, test_score = trainer.initiate_model_training(csv_path)
    print(f"Model Training complete. Train R2: {train_score}, Test R2: {test_score}")
