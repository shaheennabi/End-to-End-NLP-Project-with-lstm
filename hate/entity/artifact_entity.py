from dataclasses import dataclass



#  Data Ingestion Artifact
@dataclass
class DataIngestionArtifact:
    imbalance_data_file_path: str
    raw_data_file_path: str
