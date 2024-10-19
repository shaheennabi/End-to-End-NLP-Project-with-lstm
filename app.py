import os
import logging
import sys
import io
from fastapi import FastAPI, HTTPException, Body
import uvicorn
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from hate.pipeline.train_pipeline import TrainPipeline
from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException
from hate.constants import *

# Set up logging to handle UTF-8
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logging.info("Training completed successfully")
        return {"message": "Training Successful!"}
    except Exception as e:
        logging.exception("Error during training")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict")
async def predict_route(text):
    try:
        logging.info("entered into prediction predict_routes ")
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(text)
        logging.info(f"Prediction Response: {prediction}")  

        return {"prediction": prediction}
    

    except CustomException as ce:
        logging.error(f"Prediction error: {str(ce)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(ce)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")  # Add this line to log the error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
