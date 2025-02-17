import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process.")

        try:
            df = pd.read_csv(r"notebook/dataset/data.csv")
            logging.info(f"Data loaded successfully. Shape: {df.shape}")

            # Separate fraud and legit transactions
            legit_data = df[df['TX_FRAUD'] == 0]
            fraud_data = df[df['TX_FRAUD'] == 1]
            
            legit_data_size = len(legit_data)
            sample_size = min(14681, legit_data_size)
          
            # Undersample legit transactions
            legit_data_sampled = legit_data.sample(n=sample_size, random_state=42)

            new_df = pd.concat([legit_data_sampled, fraud_data], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
            logging.info(f"Balanced dataset created. Shape: {new_df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            new_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Splitting into train and test sets
            logging.info("Splitting data into train and test sets.")
            train_set, test_set = train_test_split(new_df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("Error in data ingestion process.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    
    modeltrainer=ModelTrainer()
    accuracy,best_model=modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print("accuracy",accuracy)
    print("best_model",best_model)
