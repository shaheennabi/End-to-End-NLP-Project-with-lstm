from dataclasses import dataclass



#  Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    imbalance_data_file_path: str
    raw_data_file_path: str



@dataclass
class DataValidationArtifact:
    validation_status:bool



@dataclass 
class DataTransformationArtifact:
    transformed_data_path: str
    


@dataclass
class ModelTrainerArtifact:
    trained_model_path:str 
    x_test_path: list
    y_test_path: list 


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool


@dataclass
class ModelPusherArtifact:
    bucket_name: str


