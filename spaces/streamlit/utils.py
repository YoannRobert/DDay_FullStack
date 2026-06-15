import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import boto3
from io import BytesIO
import os

load_dotenv()


def load_dataset_s3(file):
    """
    Load a dataset from S3 using boto3.

    Args:
        file (str): The path to the file in S3. (directory/file.csv)

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name="eu-west-3",
    )

    response = s3_client.get_object(Bucket=os.environ.get('AWS_BUCKET'), Key=file)
    df = pd.read_csv(BytesIO(response["Body"].read()))
    return df


@st.cache_data(ttl=3600)
def load_recent_dataset():
    return load_dataset_s3("dataset/training_dataset.csv")


@st.cache_data(ttl=3600)
def load_prediction_dataset():
    return load_dataset_s3("dataset/predictions.csv")