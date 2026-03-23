import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime, timedelta
import boto3
from io import BytesIO, StringIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# conversion pour prophet
def convert_df_to_prophet(df):
    df_prophet = df.copy()
    df_prophet.drop(['start_date','start_date_fr','end_date_fr'], axis=1, inplace=True)
    df_prophet.rename(columns={'end_date': 'ds', 'consumption_MW': 'y',}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], utc=True).dt.tz_convert(None)
    return df_prophet

# Création de train_set et test_set
def create_train_test(df, train_period="30 days", test_period="1 day"):
    """
    Crée une liste de train_set et test_set en partant de la fin du dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset avec une colonne 'ds' (datetime)
    train_period : str
        Durée du train_set (ex: "365 days", "2 years")
    test_period : str
        Durée du test_set (ex: "30 days", "1 month")
    
    Returns:
    --------
    tuple (train_set, test_set)
    """
    
    # Convertir les périodes en timedelta
    train_delta = pd.Timedelta(train_period)
    test_delta = pd.Timedelta(test_period)
    
    df = df.copy()
    
    # Date de fin du dataset
    end_date = df['ds'].max()
    
    # Dates du test_set
    test_start = end_date - test_delta
    test_end = end_date
        
    # Dates du train_set (avant le test_set)
    train_start = test_start - train_delta
    train_end = test_start
        
    # Filtrer les données
    train_set = df[(df['ds'] >= train_start) & (df['ds'] < train_end)].copy()
    test_set = df[(df['ds'] >= test_start) & (df['ds'] < test_end)].copy()
        
    # Ajouter seulement si les sets ne sont pas vides
    if len(train_set) > 0 and len(test_set) > 0:
        return (train_set, test_set)
    else:
        raise Exception("Pas assez de données pour créer les sets")

# train and predict 
def train_predict(train_set, test_set, regressors=[], hyperparams={}, verbose=True):

    # model
    model = Prophet(**hyperparams)
    if len(regressors) > 0:
        for regressor in regressors:
            model.add_regressor(regressor)
    model.fit(train_set)
    fields = ['ds'] + regressors
    train_pred = model.predict(train_set[fields])
    test_pred = model.predict(test_set[fields])

    # Calculate metrics
    train_mae = mean_absolute_error(train_set['y'], train_pred['yhat'])
    #test_mae = mean_absolute_error(test_set['y'], test_pred['yhat'])
    train_mape = mean_absolute_percentage_error(train_set['y'], train_pred['yhat'])
    #test_mape = mean_absolute_percentage_error(test_set['y'], test_pred['yhat'])

    if verbose:
        print(f'Train MAE: {train_mae:.2f}')
        print(f'Train MAPE: {100*train_mape:.2f}%')
    else:
        # log
        pass

    df_pred = test_pred[['ds', 'yhat']]
    #df_pred.to_csv(os.getenv('DATA_STORAGE') + 'prophet_predictions.csv', index=False)
    return df_pred

def load_training_data():
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

    response = s3_client.get_object(Bucket=os.environ.get('AWS_BUCKET'), Key=os.environ.get('TRAINING_DATA'))
    df = pd.read_csv(BytesIO(response["Body"].read()))
    return df

def save_predictions(df_pred):
    session = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name='eu-west-3')
    s3 = session.resource('s3')
    bucket = s3.Bucket(os.environ.get('AWS_BUCKET'))

    csv_buffer = StringIO()
    df_pred.to_csv(csv_buffer, index=False)

    bucket.put_object(
        Key=os.environ.get('PREDICTION_DATA'),
        Body=csv_buffer.getvalue(),
    )


######
#USE MODEL
######
df = load_training_data()

df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
df['end_date'] = pd.to_datetime(df['end_date'], utc=True)
df['start_date_fr'] = df['start_date'].dt.tz_convert('Europe/Paris')
df['end_date_fr'] = df['end_date'].dt.tz_convert('Europe/Paris')

# Last date with row not NaN
train_end_date = df[df['consumption_MW'].notna()]['end_date'].max().date()
# 30 days before
train_start_date = train_end_date - timedelta(days=30)

mask_training = (df['start_date_fr'].dt.date >= train_start_date) & (df['end_date_fr'].dt.date <= train_end_date)
df_training = df[mask_training]

df_test = df[df['start_date_fr'].dt.date == (train_end_date + timedelta(days=1))]
print("Test date:", train_end_date + timedelta(days=1))

df_train_prophet = convert_df_to_prophet(df_training)
df_test_prophet = convert_df_to_prophet(df_test)

regressors = ['T_mean_degC','T_min_degC','T_max_degC','T_std_degC','T_q1_degC','T_q3_degC']
hyperparameters = {'changepoint_prior_scale': 0.35, 'seasonality_prior_scale': 0.5}

df_pred = train_predict(df_train_prophet, df_test_prophet, regressors=regressors, hyperparams=hyperparameters, verbose=True)
save_predictions(df_pred)

