import os 
import sys
from hate.constants import *
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifact
from hate.logger import logging 
from hate.exception import HateException
from zipfile import ZipFile
from hate.configuration.s3_syncer import S3Client





class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config  = data_ingestion_config
        self.s3_bucket = S3Client()


    
    def get_data_from_s3(self) -> None: 
        logging.info("Entered the get_data_from_s3 of Data Ingestion class")
        try: 
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            self.s3_bucket.download_file(self.data_ingestion_config.BUCKET_NAME, self.data_ingestion_config.ZIP_FILE_NAME, self.data_ingestion_config.ZIP_FILE_PATH)

            logging.info("Successfully downloaded the file from S3")
            logging.info("Exited the get_data_from_s3 method of DataIngestion component")


        except Exception as e: 
            raise HateException(e,sys)



    def unzip_and_clean(self):
        logging.info("Entered te unzip_and_clean method of Data Ingestion class")
        try:
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("Exited the unzip_and_clean method of Data Ingestion class")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR


        except Exception as e: 
            raise HateException(e,sys)
    

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:  
            self.get_data_from_s3()
            logging.info("It has downlaoded the data from S3 Bucket")
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
            logging.info("Unzipped file and split into imbalance and raw csv's")

            data_ingestion_artifact = DataIngestionArtifact(imbalance_data_file_path=imbalance_data_file_path, raw_data_file_path=raw_data_file_path)
            logging.info("Exited the initiate_data_ingestion method of Data Ingestion class")

            return data_ingestion_artifact

        except Exception as e: 
            raise HateException(e,sys)
        