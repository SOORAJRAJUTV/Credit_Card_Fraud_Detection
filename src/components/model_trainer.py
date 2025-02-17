import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=500),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(algorithm="SAMME"),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [5, 10, 20, None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'criterion': ['gini', 'entropy']
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.6, 0.7, 0.8]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10]
                },
                "CatBoost Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 200]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.5],
                    'n_estimators': [50, 100, 200]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            ## Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")
            
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                   ## ✅ Print each model and its accuracy
            logging.info("Model Performance Report:")
            for model_name, accuracy in model_report.items():
                print(f"{model_name}: {accuracy:.4f}")
            logging.info(f"{model_name}: {accuracy:.4f}")

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return (accuracy,best_model)

        except Exception as e:
            raise CustomException(e, sys)
