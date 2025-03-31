import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preProcessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

logging_file_path = os.path.join(log_dir, 'data_preProcessing.log')
file_handler = logging.FileHandler(logging_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    '''
    Transforms the text into stopwords, punctuations, and stemming
    '''
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

def preProcess_df(df: pd.DataFrame, text_column: str, target_column: str) -> pd.DataFrame:
    try:
        logger.debug('starting preProcessing for DataFrame')
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logging.debug('Target column encoded')

        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates removed')

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text Column Transformed')
        return df
    except KeyError as k:
        logger.error(f'Unexpected error raised while doing the transformation as: {k}')
        raise
    except Exception as e:
        logger.error(f'Unwanted Error Occuered: {e}')
        raise

def main():
    try:
        text_column = 'text'
        target_column = 'target'

        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')

        train_preprocessed_data = preProcess_df(train_data, text_column = text_column, target_column=target_column)
        test_preprocessed_data = preProcess_df(test_data, text_column = text_column, target_column=target_column)

        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok = True)

        train_preprocessed_data.to_csv(os.path.join(data_path, 'train_preprocessed.csv'), index = False)
        test_preprocessed_data.to_csv(os.path.join(data_path, 'test_preprocessed.csv'), index = False)

        logger.debug(f'Preprocessed data saved to: {data_path}')
    
    except FileNotFoundError as fn:
        logger.error(f'File not found: {fn}')
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f'No data found: {e}')
        raise
    except Exception as e:
        logger.error(f'Failed to complete the data transformation process: {e}')
        raise

if __name__ == "__main__":
    main()