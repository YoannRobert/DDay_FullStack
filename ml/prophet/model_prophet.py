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


class ConsumptionPrediction():

    def __init__(self):
        self.s3_session = boto3.Session(
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name="eu-west-3"
        )

        self.s3_client = self.s3_session.client("s3")

        self.regressors = ['T_mean_degC','T_min_degC','T_max_degC','T_std_degC','T_q1_degC','T_q3_degC']
        self.hyperparameters = {'changepoint_prior_scale': 0.35, 'seasonality_prior_scale': 0.5}
        


    def load_dataset_s3(self, file):
        """
        Load a dataset from S3 using boto3.
        
        Args:
            file (str): The path to the file in S3. (directory/file.csv)
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        try:
            response = self.s3_client.get_object(Bucket=os.environ.get('AWS_BUCKET'), Key=file)
            df = pd.read_csv(BytesIO(response["Body"].read()))
            return df
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement des données de previsions : {e}")


    def load_predictions_data(self):
        """
        Chargement des données de prédiction (CSV) depuis bucket S3
        """
        return self.load_dataset_s3('dataset/predictions.csv')

    def load_training_data(self):
        """
        Chargement des données d'entrainement du modèle et futures pour predictions
        """
        return self.load_dataset_s3('dataset/training_dataset.csv')
    
    def create_train_prod_set(self, df):

        # conversions datetime
        df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
        df['end_date'] = pd.to_datetime(df['end_date'], utc=True)
        df['start_date_fr'] = df['start_date'].dt.tz_convert('Europe/Paris')
        df['end_date_fr'] = df['end_date'].dt.tz_convert('Europe/Paris')

        # Last date with row not NaN
        train_end_date = df[df['consumption_MW'].notna()]['end_date'].max().date()
        # 30 days before
        train_start_date = train_end_date - timedelta(days=30)

        mask_train = (df['start_date_fr'].dt.date >= train_start_date) & (df['start_date_fr'].dt.date <= train_end_date)
        df_train = df[mask_train]

        df_prod = df[df['start_date_fr'].dt.date == (train_end_date + timedelta(days=1))]

        self.train_set = self.convert_df_to_prophet(df_train)
        self.prod_set = self.convert_df_to_prophet(df_prod)

    # conversion pour prophet
    def convert_df_to_prophet(self, df):
        df_prophet = df.copy()
        df_prophet.drop(['start_date','start_date_fr','end_date_fr'], axis=1, inplace=True)
        df_prophet.rename(columns={'end_date': 'ds', 'consumption_MW': 'y',}, inplace=True)
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], utc=True).dt.tz_convert(None)
        return df_prophet


    # train and predict 
    def train_predict(self, verbose=True):

        # train model
        model = Prophet(**self.hyperparameters)
        if len(self.regressors) > 0:
            for regressor in self.regressors:
                model.add_regressor(regressor)
        model.fit(self.train_set)

        # predict
        fields = ['ds'] + self.regressors
        train_pred = model.predict(self.train_set[fields])
        prod_pred = model.predict(self.prod_set[fields])

        # Calculate metrics
        train_mae = mean_absolute_error(self.train_set['y'], train_pred['yhat'])
        #test_mae = mean_absolute_error(test_set['y'], test_pred['yhat'])
        train_mape = mean_absolute_percentage_error(self.train_set['y'], train_pred['yhat'])
        #test_mape = mean_absolute_percentage_error(test_set['y'], test_pred['yhat'])

        if verbose:
            print(f'Train MAE: {train_mae:.2f}')
            print(f'Train MAPE: {100*train_mape:.2f}%')
        else:
            # log
            pass

        self.predictions = prod_pred[['ds', 'yhat']]
        #df_pred.to_csv(os.getenv('DATA_STORAGE') + 'prophet_predictions.csv', index=False)


    def save_predictions(self):

        csv_buffer = StringIO()
        self.predictions.to_csv(csv_buffer, index=False)

        self.s3_client.put_object(
            Bucket=os.environ.get('AWS_BUCKET'),
            Key="dataset/predictions.csv",
            Body=csv_buffer.getvalue(),
        )

    def run(self):
        try:
            training_dataset = self.load_training_data()
            self.create_train_prod_set(training_dataset)
            self.train_predict()
            self.save_predictions()
        except RuntimeError as e:
            raise Exception(e)
        except Exception as e:
            raise Exception(e)



