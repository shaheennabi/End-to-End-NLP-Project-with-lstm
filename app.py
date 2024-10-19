import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import logging
import sys
import io

from hate.pipeline.train_pipeline import TrainPipeline

# Set up logging to handle UTF-8
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

obj = TrainPipeline()
obj.run_pipeline()
