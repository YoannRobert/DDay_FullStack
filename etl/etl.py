from extract import extract_datasets
from transform import transform_datasets
from load import load_dataset


def extract_transform_load(bucket : str, key : str, region : str = "eu-west-3", past_days: int = 35) -> None:
    consumption_data, weather_data = extract_datasets(past_days=past_days)
    df = transform_datasets(consumption_data=consumption_data, weather_data=weather_data)
    load_dataset(df, bucket, key, region=region)


if __name__ == "__main__":
    extract_transform_load(
        bucket = "jedha-demo-day-20260327",
        key="dataset/training_dataset.csv"
    )
