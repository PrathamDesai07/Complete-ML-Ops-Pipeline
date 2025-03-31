from fileinput import fileno
import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

logging_file_path = os.path.join(log_dir, 'Feature_engineering.log')
file_handler = logging.FileHandler(logging_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f'Safely readed the data from yaml file at location: {file_path}')
        return params
    except FileNotFoundError as e:
        logger.error(f'File not found at location: {file_path}')
        raise
    except Exception as e:
        logger.error(f'Unwanted Error Raised while reading the yaml as: {e}')
        raise

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


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    '''Apply the TFIDF method'''
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data['text'].values
        y_train = train_data['target'].values
        x_test = test_data['text'].values
        y_test = test_data['target'].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('TFIDF applied and dataTransformed')
        return train_df, test_df
    except Exception as e:
        logger.error(f'Unwanted Error occured while performing TFIDF: {e}')
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        df.to_csv(file_path, index = False)
        logger.debug(f'Data Saved to: {file_path}')
    except Exception as e:
        logger.error(f'Unexpected error occured while saving the data to path: {file_path} as {e}')
        raise

def main():
    try:
        yaml_file_path = '/teamspace/studios/this_studio/Complete-ML-Ops-Pipeline/params.yaml'
        train_data = load_data('/teamspace/studios/this_studio/Complete-ML-Ops-Pipeline/data/interim/train_preprocessed.csv')
        test_data = load_data('/teamspace/studios/this_studio/Complete-ML-Ops-Pipeline/data/interim/test_preprocessed.csv')
        
        params = load_yaml(file_path=yaml_file_path)
        max_features = params['feature_engineering']['max_features']
        # max_features = 50
        
        train_df, test_df = apply_tfidf(train_data=train_data, test_data=test_data, max_features=max_features)

        save_data(train_df, os.path.join('./data', 'processed', 'train_tfidf.csv'))
        save_data(test_df, os.path.join('./data', 'processed', 'test_tfidf.csv'))
        
    except Exception as e:
        logger.error(f'Failed to complete the feature engineering process: {e}')
        raise

if __name__ == '__main__':
    main()