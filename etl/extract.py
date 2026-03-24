from etl.get_hourly_consumption_data import fetch_consumption_data
from etl.get_hourly_weather_data_by_departments import fetch_department_weather


def extract_datasets(past_days: int = 35):
    consumption_data = fetch_consumption_data(past_days=past_days)[["start_date", "end_date", "value"]]
    weather_data = fetch_department_weather(past_days=past_days)[["Department Code", "Timestamp", "Temperature (°C)"]]
    return consumption_data, weather_data
