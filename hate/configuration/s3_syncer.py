import boto3
import os
import logging
from botocore.exceptions import NoCredentialsError, ClientError

class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self, region_name='us-east-1'):
        """
        This Class gets AWS credentials from environment variables and creates a connection with S3.
        It raises an exception when environment variables are not set.
        """
        # Singleton pattern - only initialize if not already initialized
        if S3Client.s3_resource is None or S3Client.s3_client is None:
            access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
            secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

            # Check if credentials are set in environment variables
            if access_key_id is None:
                raise Exception("Environment variable 'AWS_ACCESS_KEY_ID' is not set.")
            if secret_access_key is None:
                raise Exception("Environment variable 'AWS_SECRET_ACCESS_KEY' is not set.")

            # Create S3 resource and client
            S3Client.s3_resource = boto3.resource('s3',
                                                  aws_access_key_id=access_key_id,
                                                  aws_secret_access_key=secret_access_key,
                                                  region_name=region_name)
            S3Client.s3_client = boto3.client('s3',
                                              aws_access_key_id=access_key_id,
                                              aws_secret_access_key=secret_access_key,
                                              region_name=region_name)

        # Instance variables referring to the class-level clients/resources
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client

    def download_file(self, bucket_name, s3_file_key, local_file_name):
        """
        Downloads a file from the specified S3 bucket to the local machine.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_file_name), exist_ok=True)

        try:
            self.s3_client.download_file(bucket_name, s3_file_key, local_file_name)
            logging.info(f"File '{local_file_name}' downloaded successfully from '{bucket_name}/{s3_file_key}'")
        except boto3.exceptions.S3UploadFailedError as e:
            raise Exception(f"Error downloading file from S3: {str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")


    def upload_file(self, file_name, bucket_name, s3_file_key):
        """
        Uploads a file from the local machine to the specified S3 bucket.
        :param file_name: Path to the local file to be uploaded
        :param bucket_name: Name of the S3 bucket
        :param s3_file_key: S3 object name (key) where the file will be stored
        """
        try:
            # Check if the file exists
            if not os.path.isfile(file_name):
                raise FileNotFoundError(f"File '{file_name}' not found.")

            self.s3_client.upload_file(file_name, bucket_name, s3_file_key)
            logging.info(f"File '{file_name}' uploaded successfully to '{bucket_name}/{s3_file_key}'")
        except boto3.exceptions.S3UploadFailedError as e:
            raise Exception(f"Error uploading file to S3: {str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")
        


    def model_exists_in_s3(bucket_name: str, model_key: str) -> bool:
        s3 = boto3.client('s3')
        try:
            s3.head_object(Bucket=bucket_name, Key=model_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise