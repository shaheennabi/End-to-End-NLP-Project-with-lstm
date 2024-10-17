import os
from datetime import datetime


# Common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = "hate-speech-classification"
ZIP_FILE_NAME = 'dataset.zip'
LABEL = 'label'
TWEET = 'tweet'



"""
Data Ingestion related constants
"""

DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR  = "raw_data.csv"


""""
Data Validation related constant
"""

DATA_VALIDATION_ARTIFACTS_DIR = "DataValidationArtifacts"
DATA_VALIDATION_STATUS_DIR = "status.txt"
DATA_VALIDATION_ALL_REQUIRED_FILES = ['imbalanced_data.csv', 'raw_data.csv']
DATA_VALIDATION_ALL_IMBALANCE_REQUIRED_COLUMNS = ['id', 'label', 'tweet']
DATA_VALIDATION_ALL_RAW_REQUIRED_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']




"""
Data Transformation related constant
"""

DATA_TRANSFORMATION_ARTIFACTS_DIR  = 'DataTransformationArtifacts'
TRANSFORMED_FILE_NAME = 'final.csv'
DATA_DIR = 'data'
ID = 'id'
AXIS = 1
INPLACE = True 
DROP_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS  = 'class'