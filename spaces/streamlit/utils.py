import pandas as pd
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