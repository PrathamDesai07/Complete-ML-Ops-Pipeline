import os
import numpy as np
import pandas as pd
import  pickle
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json

from sklearn.model_selection import RandomizedSearchCV

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

logging_file_path = os.path.join(log_dir, 'Model_evaluation.log')
file_handler = logging.FileHandler(logging_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    try: 
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model Loaded Successfully")
        return model
    except FileNotFoundError as e:
        logger.error(f'File not found at path: {file_path} as {e}')
        raise
    except Exception as e:
        logger.error(f'Unwanted Exception occurred as: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from: {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Cannot Parse the csv as: {e}")
    except Exception as e:
        logger.error('Unwanted Exception raised as: {e}')
        raise

def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(x_test)
        y_pred_prob = clf.predict_proba(x_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        metrics_dict = {
            'Model': 'RandomForest',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Auc': auc
        }

        logger.debug('Model Evaluation metrics caluclated')
        return metrics_dict

    except Exception as e:
        logger.error(f'Error during model Evaluation as: {e}')
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f'Metrics saved at path: {file_path}')
    except Exception as e:
        logger.error(f'Unwanted error raised as: {e}')
        raise

def main():
    try:
        model_path = "/teamspace/studios/this_studio/Complete-ML-Ops-Pipeline/models/model.pkl"
        test_data_path = "/teamspace/studios/this_studio/Complete-ML-Ops-Pipeline/data/processed/test_tfidf.csv"
        metrics_path = "./reports/metrics.json"

        clf = load_model(model_path)
        test_data = load_data(test_data_path)

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, x_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        logger.error(f'Unwanted error raise as {e}')
        raise

if __name__ == "__main__":
    main()