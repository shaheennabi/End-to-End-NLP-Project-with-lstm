import os
import sys
import keras
import pickle
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from keras.utils import pad_sequences
from hate.configuration.s3_syncer import S3Client
from hate.components.data_transformation import DataTransformation
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifact
from hate.entity.artifact_entity import DataValidationArtifact

class PredictionPipeline:
    def __init__(self): 
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join("artifacts", "PredicModel")
        self.tokenizer_path = "tokenizer.pickle" 
        self.s3 = S3Client()
        self.data_transformation = DataTransformation(
            data_transformation_config=DataTransformationConfig,  
            data_ingestion_artifact=DataIngestionArtifact,        
            data_validation_artifact=DataValidationArtifact       
        )

    def get_model_from_s3(self) -> str:
        """
        Method Name : get_model_from_s3
        Description :  This method downloads the model from S3 storage.
        Output      : Path of the best model
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")

        try: 
            os.makedirs(self.model_path, exist_ok=True)
            model_file_path = os.path.join(self.model_path, self.model_name)  # Define full model path
            
            # Check if the model file already exists
            if os.path.exists(model_file_path):
                logging.info(f"Model already exists at: {model_file_path}")
                return model_file_path  # Return the existing model path

            # Log the model path before downloading
            logging.info(f"Downloading model to: {model_file_path}")
            self.s3.download_file(self.bucket_name, self.model_name, model_file_path)

            logging.info(f"Model downloaded from S3: {model_file_path}")
            return model_file_path

        except Exception as e: 
            raise CustomException(e, sys) from e 

    def load_tokenizer(self):
        """
        Method to load the tokenizer.
        """
        try:
            # Ensure tokenizer path is correct and file exists
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {self.tokenizer_path}")
            
            with open(self.tokenizer_path, 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            logging.info("Tokenizer loaded successfully")
            return load_tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, best_model_path, text):
        """
        Method to perform prediction on the input text.
        """
        logging.info("Entered predict method of PredictionPipeline class")

        try:
            # Load the model
            load_model = keras.models.load_model(best_model_path)

            # Load tokenizer
            load_tokenizer = self.load_tokenizer()

            # Preprocess the input text
            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]  

            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)

            # Perform prediction
            pred = load_model.predict(padded)

            logging.info(f"Prediction result: {pred}")
            if pred[0] > 0.5: 
                return {"label": "hate and abusive", "confidence": float(pred[0])}
            else:
                return {"label": "no hate", "confidence": float(1 - pred[0])}
                
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text):
        """
        This method runs the entire prediction pipeline.
        """
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            best_model_path = self.get_model_from_s3()  # Ensure the model is downloaded
            predicted_text = self.predict(best_model_path, text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        
        except Exception as e:
            raise CustomException(e, sys) from e
