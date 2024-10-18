import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.constants import *
from hate.configuration.s3_syncer import S3Client
from sklearn.metrics import confusion_matrix
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataTransformationArtifact


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
        """
        :param model_evaluation_config: Configuration for model evaluation.
        :param model_trainer_artifacts: Output reference of model trainer artifact stage.
        :param data_transformation_artifacts: Reference to data transformation stage.
        """
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifacts = data_transformation_artifact
        self.s3 = S3Client()

    def get_best_model_from_s3(self) -> str:
        """
        Fetch best model from S3 and store it inside the best model directory path.
        :return: Path of the best model.
        """
        try:
            logging.info("Entered the get_best_model_from_s3 of Model Evaluation class")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            self.s3.download_file(self.model_evaluation_config.BUCKET_NAME,
                                  self.model_evaluation_config.MODEL_NAME,
                                  self.model_evaluation_config.BEST_MODEL_DIR_PATH)

            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluation_config.MODEL_NAME)

            logging.info("Exited the get_best_model_from_s3 method of model evaluation class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(self):
        """
        Evaluate the currently trained model or the best model from S3 storage using the test dataset.

        :return: Accuracy of the model.
        """
        try:
            logging.info("Entering the evaluate function of Model Evaluation class")

            # Load test data
            x_test = pd.read_csv(self.model_trainer_artifact.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifact.y_test_path, index_col=0)

            # Load tokenizer
            with open('../../tokenizer/tokenizer.pickle', 'rb') as handle:  # Update the path as necessary
                tokenizer = pickle.load(handle)

            # Load trained model
            load_model = keras.models.load_model(self.model_trainer_artifact.trained_model_path)

            # Preprocess test data
            x_test = x_test['tweet'].astype(str).squeeze()
            y_test = y_test.squeeze()

            # Convert to sequences
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            # Evaluate model
            loss, accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"The test accuracy is {accuracy}")

            # Generate predictions and compute confusion matrix
            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = [1 if prediction[0] >= 0.5 else 0 for prediction in lstm_prediction]

            cm = confusion_matrix(y_test, res)
            logging.info(f"The confusion_matrix is {cm}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiate all steps of the model evaluation.

        :return: Returns model evaluation artifact.
        """
        logging.info("Initiate Model Evaluation")
        try:
            logging.info("Loading currently trained model")
            trained_model = keras.models.load_model(self.model_trainer_artifact.trained_model_path)

            # Evaluate the trained model
            trained_model_accuracy = self.evaluate()

            logging.info("Fetch best model from S3 storage")
            best_model_path = self.get_best_model_from_s3()

            logging.info("Check if the best model is present in the S3 storage or not")
            if not os.path.isfile(best_model_path):
                is_model_accepted = True
                logging.info("Best model not found in S3; trained model accepted.")
            else:
                logging.info("Load best model fetched from S3 storage")
                best_model = keras.models.load_model(best_model_path)
                
                # Evaluate best model
                best_model_accuracy = self.evaluate()

                logging.info("Comparing accuracy between best_model_accuracy and trained_model_accuracy")
                if trained_model_accuracy > best_model_accuracy:
                    is_model_accepted = True
                    logging.info("Trained model has better accuracy; it is accepted.")
                else:
                    is_model_accepted = False
                    logging.info("Best model has better accuracy; trained model not accepted.")

            model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifact")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
