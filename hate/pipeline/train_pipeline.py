import sys
from hate.logger import logging
from hate.exception import HateException
from hate.exception import CustomException
from hate.components.data_ingestion import DataIngestion
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifact
from hate.components.data_validation import DataValidation
from hate.entity.config_entity import DataValidationConfig
from hate.entity.artifact_entity import DataValidationArtifact
from hate.components.data_transformation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataTransformationArtifact
from hate.components.model_trainer import ModelTrainer
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifact
from hate.components.model_evaluation import ModelEvaluation
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifact
from hate.components.model_pusher import ModelPusher
from hate.entity.artifact_entity import ModelPusherArtifact
from hate.entity.config_entity import ModelPusherConfig


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

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

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("Entering the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,  
                data_validation_config=self.data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact

        except Exception as e:
            raise HateException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,  
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifacts

        except Exception as e:
            raise HateException(e, sys)

    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("Entered the start_model_trainer method of TrainPipeline class")
        try:
            model_trainer = ModelTrainer(data_transformation_artifacts=data_transformation_artifacts, 
                                         model_trainer_config=self.model_trainer_config)
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifacts

        except Exception as e:
            raise HateException(e, sys)

    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")
        try: 
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact,
                  
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Exited the start_model_evaluation method of TrainPipeline class")
            return model_evaluation_artifact

        except Exception as e: 
            raise CustomException(e, sys)

    def start_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered the start_model_pusher method of TrainPipeline class")
        try: 
            model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Initiated the model pusher")
            logging.info("Exited the start_model_pusher method of TrainPipeline class")
            return model_pusher_artifact

        except Exception as e: 
            raise CustomException(e, sys)

    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact, 
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifacts=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact
            )

            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            else: 
                model_pusher_artifact = self.start_model_pusher()
                logging.info(f"Model pusher artifact: {model_pusher_artifact}")

            logging.info("Pipeline completed successfully")
            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise HateException(e, sys)
