from dataclasses import dataclass
from hate.constants import *
import os




@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME = BUCKET_NAME
        self.ZIP_FILE_NAME = ZIP_FILE_NAME

        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATA_INGESTION_IMBALANCE_DATA_DIR)
        self.NEW_DATA_ARTIFACTS_DIR: str = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATA_INGESTION_RAW_DATA_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,self.ZIP_FILE_NAME)




@dataclass
class DataValidationConfig:
    def __init__(self):
        self.DATA_VALIDATION_ARTIFACTS_DIR: str = os.path.join(ARTIFACTS_DIR, DATA_VALIDATION_ARTIFACTS_DIR)
        self.STATUS_FILE_DIR: str = os.path.join(self.DATA_VALIDATION_ARTIFACTS_DIR, DATA_VALIDATION_STATUS_DIR)
        self.REQUIRED_FILES_LIST = DATA_VALIDATION_ALL_REQUIRED_FILES
        self.IMBALANCE_COLUMNS_LIST = DATA_VALIDATION_ALL_IMBALANCE_REQUIRED_COLUMNS
        self.RAW_COLUMNS_LIST = DATA_VALIDATION_ALL_RAW_REQUIRED_COLUMNS


        
    