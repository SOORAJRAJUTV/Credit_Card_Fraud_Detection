import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading Model and Preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading Model and Preprocessor")

            # Transform the input data
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class takes user input transaction data and converts it into a pandas DataFrame
    that can be used for predictions.
    """

    def __init__(
        self,
        customer_id: str,
        terminal_id: str,
        tx_amount: float,
        tx_hour: int,
        tx_day: int,
        tx_day_of_week: int,
        tx_month: int,
        is_night_tx: int,
        is_weekend_tx: int
    ):
        self.customer_id = customer_id
        self.terminal_id = terminal_id
        self.tx_amount = tx_amount
        self.tx_hour = tx_hour
        self.tx_day = tx_day
        self.tx_day_of_week = tx_day_of_week
        self.tx_month = tx_month
        self.is_night_tx = is_night_tx
        self.is_weekend_tx = is_weekend_tx

    def get_data_as_data_frame(self):
        """
        Converts user input transaction data into a DataFrame format for prediction.
        """
        try:
            custom_data_input_dict = {
                "CUSTOMER_ID": [self.customer_id],
                "TERMINAL_ID": [self.terminal_id],
                "TX_AMOUNT": [self.tx_amount],
                "TX_HOUR": [self.tx_hour],
                "TX_DAY": [self.tx_day],
                "TX_DAY_OF_WEEK": [self.tx_day_of_week],
                "TX_MONTH": [self.tx_month],
                "IS_NIGHT_TX": [self.is_night_tx],
                "IS_WEEKEND_TX": [self.is_weekend_tx],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
