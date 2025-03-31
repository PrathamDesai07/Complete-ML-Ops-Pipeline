import os
import numpy as np
import pandas as pd
import  pickle
import logging
from sklearn.ensemble import RandomForestClassifier

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

logging_file_path = os.path.join(log_dir, 'Model_training.log')
file_handler = logging.FileHandler(logging_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    '''
    Loads the data from a specific url/path
    '''
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace = True)
        logger.debug(f'CSV file loaded from path: {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'failed to parse the csv file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unwanted Error occured while loading the data from URL/PATH: {file_path} as {e}')
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    '''
    Trains the RandomForest Model
    '''
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X-train and Y-train must be same")
        logger.debug(f'Initializing the training of RandomForest with hyper params: {params}')
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        logger.debug(f'Model Training stated with {x_train.shape[0]} samples')
        clf.fit(x_train, y_train)
        logger.debug('Training done')
        return clf
    except ValueError as v:
        logger.error(f'Value Error Raised during the model Training as: {v}')
        raise
    except Exception as e:
        logger.error(f'Unwanted Error Occured while training the model: {e}')
        raise

def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok =True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f'file saved to path {file_path}')
    except FileNotFoundError as e:
        logger.error(f'File path not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error occured while saving the model as: {e}')
        raise

def main():
    try: 
        params = {'n_estimators': 25, 'random_state':2}
        train_data = load_data('/teamspace/studios/this_studio/Complete-ML-Ops-Pipeline/data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(x_train, y_train, params)
        model_save_path = './models/model.pkl'
        save_model(clf, model_save_path)
    except Exception as e:
        logger.error(f'Unwanted Error occured as: {e}')
        raise

if __name__ == "__main__":
    main()