import sys
from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.s3_syncer import S3Client
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifact


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        """
        param model_pusher_config: Configuration for model pusher
        """
        self.model_pusher_config = model_pusher_config
        self.s3 = S3Client()

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method name: initiate_model_pusher
        Description: This method initiates model pusher.

        Output: Model pusher artifact
        """

        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            # Uploading the model to S3 storage with correct argument order
            self.s3.upload_file(self.model_pusher_config.TRAINED_MODEL_PATH,
                                self.model_pusher_config.BUCKET_NAME,
                                self.model_pusher_config.MODEL_NAME)

            logging.info("Model uploaded to S3 storage successfully.")
            
            # Saving the model pusher artifacts
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.BUCKET_NAME)

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
