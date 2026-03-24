import uvicorn
from fastapi import FastAPI
from etl.etl import *
from ml.prophet.model_prophet import ConsumptionPrediction
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/predict")
async def index():

    try:
        extract_transform_load(bucket=os.environ.get('AWS_BUCKET'), key=os.environ.get('TRAINING_DATA'))
    except:
        return 'There was an error during ETL'

    try:
        predictor = ConsumptionPrediction()
        predictor.run()
    except:
        return 'There was an error while predicting'

    return 'predicition.csv created on S3 bucket'

uvicorn.run(app, host="0.0.0.0", port=4000)