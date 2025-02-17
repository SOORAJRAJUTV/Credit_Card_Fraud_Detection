import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")




class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()





    def time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts meaningful time-based features from TX_DATETIME."""
        try:
            df = df.copy()
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
            df['TX_HOUR'] = df['TX_DATETIME'].dt.hour # dt: date time Aceesor
            df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek
            df['TX_DAY'] = df['TX_DATETIME'].dt.day
            df['TX_MONTH'] = df['TX_DATETIME'].dt.month
            df['IS_NIGHT_TX'] = (df['TX_HOUR'] < 6).astype(int)
            df['IS_WEEKEND_TX'] = (df['TX_DAY_OF_WEEK'] >= 5).astype(int)
            df.drop(columns=['TX_DATETIME'], inplace=True)
            return df
        
        except Exception as e:
            raise CustomException(e, sys)






    def top_n_encoding(self, df: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
        """Encodes high-cardinality categorical variables using Top-N encoding."""
        try:
            df = df.copy()
            top_customers = df['CUSTOMER_ID'].value_counts().nlargest(top_n).index
            df['CUSTOMER_ID'] = np.where(df['CUSTOMER_ID'].isin(top_customers), df['CUSTOMER_ID'], 'OTHER')
            
            top_terminals = df['TERMINAL_ID'].value_counts().nlargest(top_n).index
            df['TERMINAL_ID'] = np.where(df['TERMINAL_ID'].isin(top_terminals), df['TERMINAL_ID'], 'OTHER')
            return df
        
        except Exception as e:
            raise CustomException(e, sys)






    def get_data_transformer_object(self):
        """Creates preprocessing pipelines for numerical and categorical features."""
        try:
            numerical_columns = ['TX_AMOUNT', 'TX_HOUR', 'TX_DAY', 'TX_MONTH', 'TX_DAY_OF_WEEK']
            categorical_columns = ['CUSTOMER_ID', 'TERMINAL_ID', 'IS_NIGHT_TX', 'IS_WEEKEND_TX']
            
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            
            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)






    def initiate_data_transformation(self, train_path: str, test_path: str):
        """Handles full data transformation pipeline including feature engineering and encoding."""
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Applying feature engineering")
            train_df = self.time_features(train_df)
            test_df = self.time_features(test_df)
            
            train_df = self.top_n_encoding(train_df)
            test_df = self.top_n_encoding(test_df)
            
            drop_columns = ["TX_FRAUD_SCENARIO", "TRANSACTION_ID", "TX_TIME_SECONDS", "TX_TIME_DAYS"]
            
            X_train = train_df.drop(columns=["TX_FRAUD"] + drop_columns, errors='ignore')
            y_train = train_df["TX_FRAUD"]
            
            X_test = test_df.drop(columns=["TX_FRAUD"] + drop_columns, errors='ignore')
            y_test = test_df["TX_FRAUD"]
            
            preprocessor = self.get_data_transformer_object()
            
            logging.info("Fitting and transforming data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]
            
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            logging.info("Data transformation completed successfully")
            
            return (
                train_arr,
                test_arr,
                self.config.preprocessor_obj_file_path,
            )




        except Exception as e:
            raise CustomException(e, sys)


