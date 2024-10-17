import os
import sys
import pandas as pd
from pandas import DataFrame
from hate.logger import logging
from hate.exception import HateException
from hate.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from hate.entity.config_entity import DataValidationConfig
from hate.constants import (DATA_VALIDATION_ALL_IMBALANCE_REQUIRED_COLUMNS, DATA_VALIDATION_ALL_RAW_REQUIRED_COLUMNS)

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise HateException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Read CSV file and return a DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HateException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame, expected_columns: list) -> bool:
        actual_columns = list(dataframe.columns)
        return set(actual_columns) == set(expected_columns)


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Validate both the 'imbalanced_data.csv' and 'raw_data.csv' files for the required columns.
        """
        try:
            # Read data
            imbalance_data = DataValidation.read_data(file_path=self.data_ingestion_artifact.imbalance_data_file_path)
            raw_data = DataValidation.read_data(file_path=self.data_ingestion_artifact.raw_data_file_path)

            # Validate columns for imbalanced data
            imbalance_status = self.validate_number_of_columns(
                dataframe=imbalance_data, 
                expected_columns=self.data_validation_config.IMBALANCE_COLUMNS_LIST
            )

            # Validate columns for raw data
            raw_status = self.validate_number_of_columns(
                dataframe=raw_data, 
                expected_columns=self.data_validation_config.RAW_COLUMNS_LIST
            )

            # Overall validation status
            validation_status = imbalance_status and raw_status

            # Save validation status
            os.makedirs(self.data_validation_config.DATA_VALIDATION_ARTIFACTS_DIR, exist_ok=True)
            with open(self.data_validation_config.STATUS_FILE_DIR, 'w') as f:
                f.write(f"Imbalance Data Validation: {imbalance_status}\n")
                f.write(f"Raw Data Validation: {raw_status}\n")
                f.write(f"Overall Validation Status: {validation_status}\n")

            # Create DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(validation_status=validation_status)

            return data_validation_artifact

        except Exception as e:
            raise HateException(e, sys)
