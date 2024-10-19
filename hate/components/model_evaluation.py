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
from hate.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig, model_trainer_artifact: ModelTrainerArtifact):
        """
        :param model_evaluation_config: Configuration for model evaluation.
        :param model_trainer_artifacts: Output reference of model trainer artifact stage.
        """
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifact = model_trainer_artifact
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
            logging.info("Loading test data")
            x_test = pd.read_csv(self.model_trainer_artifact.x_test_path, header=0)
            y_test = pd.read_csv(self.model_trainer_artifact.y_test_path, header=0)


            # Load tokenizer
            logging.info("Entering tokenizer loading")
            with open('C:/Users/mailm/downloads/projects/End-to-End-NLP-Project-with-lstm/tokenizer/tokenizer.pickle', 'rb') as handle: 
                tokenizer = pickle.load(handle)
            logging.info("Exiting loading tokenizer process")

            # Load trained model
            load_model = keras.models.load_model(self.model_trainer_artifact.trained_model_path)

            # Preprocess test data
            logging.info("Preprocessing test data")
            print(x_test.columns)  # Check the column names

            x_test = x_test['tweet'].tolist()  
            x_test = [str(text) for text in x_test]


            # Convert x_test to strings
            x_test = [str(text) for text in x_test]
            logging.info(f"x_test converted to strings: {x_test[:5]}")  # Log first 5 for verification
            
            # Ensure y_test is a single-column DataFrame
            if y_test.shape[1] == 1:  # Check if there is only one column
                y_test = y_test.squeeze()  # Convert to Series
            else:
                logging.error("y_test has more than one column, which is unexpected.")
                raise ValueError("y_test should have a single column for labels.")

            # Check the shape and contents of y_test
            logging.info(f"y_test shape: {y_test.shape}, sample: {y_test.head()}")

            logging.info("Converting x_test to sequences")
            # Convert to sequences
            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            logging.info("Starting model evaluation")
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
        logging.info("Initiating Model Evaluation")
        try:
            logging.info("Loading currently trained model")
            trained_model = keras.models.load_model(self.model_trainer_artifact.trained_model_path)

            # Evaluate the trained model
            trained_model_accuracy = self.evaluate()

            logging.info("Checking if best model exists in S3 storage")
            s3_bucket_name = self.model_evaluation_config.BUCKET_NAME
            s3_model_key = self.model_evaluation_config.MODEL_NAME

            # Check if the model exists in S3
            if not self.s3.model_exists_in_s3(s3_bucket_name, s3_model_key):
                logging.info("No model found in S3. Accepting trained model as best model.")
                is_model_accepted = True
            else:
                logging.info("Best model exists in S3. Downloading and evaluating.")
                # Download the model from S3 to a local path
                local_best_model_path = self.model_evaluation_config.BEST_MODEL_DIR_PATH
                self.get_best_model_from_s3(s3_bucket_name, s3_model_key, local_best_model_path)
                
                best_model = keras.models.load_model(local_best_model_path)
                best_model_accuracy = self.evaluate()

                logging.info("Comparing best model and trained model accuracies")
                if best_model_accuracy > trained_model_accuracy:
                    is_model_accepted = False
                    logging.info("Trained model rejected. Best model in S3 is better.")
                else:
                    is_model_accepted = True
                    logging.info("Trained model accepted. It is better than the best model in S3.")

            model_evaluation_artifact = ModelEvaluationArtifact(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifact")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
