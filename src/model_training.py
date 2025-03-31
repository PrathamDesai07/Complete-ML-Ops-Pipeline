import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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