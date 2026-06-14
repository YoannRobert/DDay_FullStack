import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from etl.etl import extract_transform_load
from ml.prophet.model_prophet import ConsumptionPrediction

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/predict")
async def predict():
    try:
        extract_transform_load(
            bucket=os.environ["AWS_BUCKET"],
            key=os.environ["TRAINING_DATA"],
        )
    except Exception as exc:
        # logger.exception emits the full stack trace to the API logs,
        # while the HTTP response stays compact for the client.
        logger.exception("ETL step failed")
        raise HTTPException(
            status_code=500,
            detail=f"ETL step failed: {type(exc).__name__}: {exc}",
        )

    try:
        ConsumptionPrediction().run()
    except Exception as exc:
        logger.exception("Prediction step failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction step failed: {type(exc).__name__}: {exc}",
        )

    return {"status": "ok", "message": "predictions.csv updated on S3 bucket"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)