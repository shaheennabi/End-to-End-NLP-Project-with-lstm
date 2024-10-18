import os
import re
import sys
import string
import pandas as pd
import nltk
from  nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from sklearn.model_selection  import train_test_split
from hate.logger import logging
from  hate.exception import HateException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifact
from hate.entity.artifact_entity import DataTransformationArtifact
from hate.entity.artifact_entity import DataValidationArtifact


class DataTransformation:
    def  __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact


    
    def imbalance_data_cleaning(self):
        
        try: 

            logging.info("Entered into the imbalance_data_cleaning function")
            imbalance_data=pd.read_csv(self.data_ingestion_artifact.imbalance_data_file_path)
            imbalance_data.drop(self.data_transformation_config.ID,axis=self.data_transformation_config.AXIS,  inplace= self.data_transformation_config.INPLACE)
            logging.info(f"Exited  the imbalance data_cleaning function and returned imbalance data {imbalance_data}")
            return imbalance_data
        
        except Exception as e:
            raise HateException(e,sys)
        


    def raw_data_cleaning(self):

        try:
            logging.info("Entered the raw_data_cleaning method of Data Transformation class")
            raw_data = pd.read_csv(self.data_ingestion_artifact.raw_data_file_path)
            raw_data.drop(self.data_transformation_config.DROP_COLUMNS, axis=self.data_transformation_config.AXIS, inplace=self.data_transformation_config.INPLACE)

            raw_data.loc[raw_data[self.data_transformation_config.CLASS] == 0, self.data_transformation_config.CLASS] = 1

            # replace the value of 0 to 1
            raw_data[self.data_transformation_config.CLASS].replace({0:1},inplace=True)

            # Let's replace the value of 2 to 0.
            raw_data[self.data_transformation_config.CLASS].replace({2:0}, inplace = True)

            # Let's change the name of the 'class' to label
            raw_data.rename(columns={self.data_transformation_config.CLASS:self.data_transformation_config.LABEL},inplace =True)
            logging.info(f"Exited the raw_data_cleaning function and returned the raw_data {raw_data}")
            return raw_data
        except Exception as e:
            raise HateException(e,sys)
        

    def concat_dataframe(self):

        try:
            logging.info("Entered into the concat_dataframe function")
            # Let's concatinate both the data into a single data frame.
            frame = [self.raw_data_cleaning(), self.imbalance_data_cleaning()]
            df = pd.concat(frame)
            print(df.head())
            logging.info(f"returned the concatinated dataframe {df}")
            return df

        except Exception as e:
            raise HateException(e, sys)   

    

    def concat_data_cleaning(self,words):

        try:
            # Initialize stemmer and stopwords
            stemmer = PorterStemmer()
            stopword = set(stopwords.words('english'))

            # Convert to string and lowercase
            words = str(words).lower()

            # Remove emojis and special characters
            words = re.sub(r'[^\x00-\x7F]+', '', words)

            # Remove URLs
            words = re.sub('https?://\S+|www\.\S+', '', words)

            # Remove HTML tags
            words = re.sub('<.*?>', '', words)

            # Remove punctuation
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)

            # Remove newlines
            words = re.sub('\n', '', words)

            # Remove words containing digits
            words = re.sub('\w*\d\w*', '', words)

            # Remove stopwords
            words = [word for word in words.split() if word not in stopword]
            words = " ".join(words)

            # Apply stemming
            words = [stemmer.stem(word) for word in words.split()]
            words = " ".join(words)

            return words
        
        except Exception as e: 
            raise HateException(e,sys) 
        


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Entered the initiate_data_transformation method of Data transformation class")

                # Perform data cleaning and transformation steps
                self.imbalance_data_cleaning()
                self.raw_data_cleaning()
                df = self.concat_dataframe()

                # Apply the cleaning function to the TWEET column
                df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.concat_data_cleaning)

                # Create the directory for storing transformed data if it doesn't exist
                os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)

                # Save the transformed data to a CSV file
                df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)

                # Create DataTransformationArtifact
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
                )
                
                logging.info("Returning the DataTransformationArtifacts")
                return data_transformation_artifact
            
            else:
                # Handle the case where data validation fails
                logging.error("Data validation failed, skipping data transformation.")
                raise CustomException("Data validation failed. Transformation cannot proceed.", sys)

        except Exception as e:
            raise HateException(e, sys) 
