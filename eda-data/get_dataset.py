import pandas as pd
from get_hourly_consumption_data import fetch_consumption_data
from get_hourly_weather_data_by_departments import fetch_department_weather


def get_dataset(past_days: int = 35):
    consumption_data = fetch_consumption_data(past_days=past_days)[["start_date", "end_date", "value"]]
    weather_data = fetch_department_weather(past_days=past_days)[["Department Code", "Timestamp", "Temperature (°C)"]]

    consumption_data.rename(columns={"Timestamp": "end_date"}, inplace=True)
    weather_data.rename(columns={"Department Code": "dept_code", "Temperature (°C)": "temperature"}, inplace=True)

    consumption_data["start_date"] = pd.to_datetime(consumption_data["start_date"], utc=True).dt.tz_convert(None)
    consumption_data["end_date"] = pd.to_datetime(consumption_data["end_date"], utc=True).dt.tz_convert(None)
    weather_data["Timestamp"] = pd.to_datetime(weather_data["Timestamp"])

    weather_data_grouped_by_timestamp = weather_data.groupby('Timestamp')['temperature'].agg(
        [
            'mean',
            'min',
            'max',
            'median',
            'std',
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75))
        ]
    ).reset_index()

    df = pd.merge(consumption_data, weather_data_grouped_by_timestamp, left_on='end_date', right_on="Timestamp", how='right')

    first_valid = df["value"].first_valid_index()
    if first_valid is not None:
        df = df.loc[first_valid:]

    df.drop(columns=['end_date'], inplace=True, axis=1)
    df.rename(columns={"Timestamp": "end_date"}, inplace=True)
    df["start_date"] = df["end_date"] - pd.Timedelta(hours=1)
    columns = df.columns.tolist()
    columns.remove("value")
    columns.insert(2, "value")
    df = df.loc[:, columns]

    df.rename(
        columns={
            "value": "consumption_MW",
            "mean": "T_mean_degC",
            "min": "T_min_degC",
            "max": "T_max_degC",
            "median": "T_median_degC",
            "std": "T_std_degC",
            "q1": "T_q1_degC",
            "q3": "T_q3_degC",
        },
        inplace=True)

    return df.reset_index(drop=True)
