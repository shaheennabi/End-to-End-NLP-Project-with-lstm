from hate.logger import logging
from hate.exception import HateException
import os
from hate.configuration.s3_syncer import S3Client
import sys

logging.info("Entered S3 client")

try:
    # Initialize S3 client
    s3_client = S3Client()

    # Define bucket, S3 key, and local file path
    bucket_name = "hate-speech-classification"
    s3_file_key = "dataset.zip"
    local_file_name = "downloads/dataset.zip"

    # Download the file from S3
    s3_client.download_file(bucket_name, s3_file_key, local_file_name)

except Exception as e:
    raise HateException(e, sys)

logging.info("Leaving after downloading the file")
