import  sys
from hate.logger import logging
from hate.exception import HateException
from hate.components.data_ingestion import DataIngestion
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        



    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try: 
            logging.info("Getting the data from S3Bucket")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the imbalanced and raw csv's from S3 bucket")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        
        except Exception as e: 
            raise HateException(e, sys)
        

    def run_pipeline(self):
        logging.info("Entered the run_pipeline method TrainPipeline class")
        try: 
            data_ingestion_artifact = self.start_data_ingestion()

            logging.info("Exited the run_pipeline method of TrainPipeline class")
        except Exception as e: 
            raise HateException(e,sys)