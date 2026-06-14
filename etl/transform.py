import pandas as pd

def transform_datasets(consumption_data, weather_data):
    weather_data.rename(
        columns={"Department Code": "dept_code", "Temperature (°C)": "temperature"},
        inplace=True
    )

    # Keep everything tz-aware UTC in memory
    consumption_data["start_date"] = pd.to_datetime(consumption_data["start_date"], utc=True)
    consumption_data["end_date"] = pd.to_datetime(consumption_data["end_date"], utc=True)
    # Open-Meteo returns ISO strings without offset (timezone=UTC requested).
    # utc=True re-tags them explicitly as UTC.
    weather_data["Timestamp"] = pd.to_datetime(weather_data["Timestamp"], utc=True)

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

    # Merge on the START of the hour: RTE labels its period by end_date,
    # Open-Meteo labels it by start (Timestamp). Aligning on start_date
    # is the only way to keep consumption and weather on the same hour.
    df = pd.merge(
        consumption_data[['start_date', 'value']],
        weather_data_grouped_by_timestamp,
        left_on='start_date', right_on='Timestamp',
        how='right'
    )

    first_valid = df["value"].first_valid_index()
    if first_valid is not None:
        df = df.loc[first_valid:]

    df.drop(columns=['start_date'], inplace=True)
    df.rename(columns={"Timestamp": "start_date"}, inplace=True)
    df["end_date"] = df["start_date"] + pd.Timedelta(hours=1)

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
        inplace=True
    )

    # Reorder so start_date comes before end_date (after merge end_date is last)
    cols = df.columns.tolist()
    cols.remove("end_date")
    start_idx = cols.index("start_date")
    cols.insert(start_idx + 1, "end_date")
    df = df.loc[:, cols]

    return df.reset_index(drop=True)