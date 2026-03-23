import boto3
import os

from dotenv import load_dotenv
from io import StringIO


def get_credentials():
    load_dotenv()
    key_id = os.getenv("AWS_ACCESS_KEY_ID")
    access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return key_id, access_key


def load_dataset(df, bucket, key, region="eu-west-3"):

    key_id, access_key = get_credentials()

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key,
        region_name=region,
    )

    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue().encode("utf-8"))
