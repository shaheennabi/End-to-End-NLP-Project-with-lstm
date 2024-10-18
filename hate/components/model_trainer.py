import os
import sys
import pandas as pd
import pickle
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from hate.ml.model import ModelArchitecture

class ModelTrainer:

    def __init__(self, data_transformation_artifacts: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def splitting_data(self, csv_path: str) -> tuple:
        try:
            logging.info("Entered the splitting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into x and y")
            x = df[self.model_trainer_config.TWEET]
            y = df[self.model_trainer_config.LABEL]

            logging.info("Applying train_test_split on the data")
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, 
                test_size=0.3, 
                random_state=self.model_trainer_config.RANDOM_STATE
            )
            logging.info(f"Training set size: {len(x_train)} | {len(y_train)}")
            logging.info(f"Testing set size: {len(x_test)} | {len(y_test)}")
            logging.info(f"Types - X_train: {type(x_train)}, Y_train: {type(y_train)}")
            logging.info("Exited the splitting_data function")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenizing(self, x_train):
        try: 
            logging.info("Applying tokenization on the data")
            # Ensure all elements in x_train are strings
            x_train = [str(text) for text in x_train]  # Convert all to strings
            
            # Initialize the Tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(x_train)

            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"Converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"The sequence matrix is: {sequences_matrix}")

            return sequences_matrix, tokenizer
        except Exception as e: 
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            # Split the data into training and test sets
            x_train, x_test, y_train, y_test = self.splitting_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path
            )
            
            # Initialize the model architecture with the configuration
            model_architecture = ModelArchitecture(self.model_trainer_config)   
            model = model_architecture.get_model()

            logging.info(f"X_train size: {len(x_train)}")
            logging.info(f"X_test size: {len(x_test)}")

            # Tokenize the input data
            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info("Starting model training")
            # Train the model
            model.fit(sequences_matrix, y_train, 
                      batch_size=self.model_trainer_config.BATCH_SIZE, 
                      epochs=self.model_trainer_config.EPOCH, 
                      validation_split=self.model_trainer_config.VALIDATION_SPLIT)
            logging.info("Model training finished")

            # Save the tokenizer to the specified directory
            os.makedirs(self.model_trainer_config.TOKENIZER_DIR, exist_ok=True)
            tokenizer_path = os.path.join(self.model_trainer_config.TOKENIZER_DIR, 'tokenizer.pickle')
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


            # Create directories and save the model
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)
            trained_model_path = os.path.join(self.model_trainer_config.TRAINED_MODEL_PATH, 'model.h5')
            logging.info("Saving the model")
            model.save(trained_model_path)

              

            # Save train and test data
            pd.DataFrame(x_train).to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH, index=False)
            pd.DataFrame(x_test).to_csv(self.model_trainer_config.X_TEST_DATA_PATH, index=False)
            pd.DataFrame(y_test).to_csv(self.model_trainer_config.Y_TEST_DATA_PATH, index=False)

            # Return the model trainer artifacts
            model_trainer_artifacts = ModelTrainerArtifact(
                trained_model_path=trained_model_path,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH
            )
            logging.info("Returning the ModelTrainerArtifact")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
