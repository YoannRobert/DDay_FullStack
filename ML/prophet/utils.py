import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# conversion pour prophet
def convert_df_to_prophet(df):
    df_prophet = df.copy()
    df_prophet.drop('start_date', axis=1, inplace=True)
    df_prophet.rename(columns={'end_date': 'ds', 'value': 'y',}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], utc=True).dt.tz_convert(None)
    return df_prophet

# Création de train_set et test_set
def create_train_test(df, train_period, test_period, period="3 days", n_iter=1):
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
    n_iter : int
        Nombre de paires train/test à créer
    period : str
        Décalage temporel entre chaque itération (ex: "3 days", "1 week")
    
    Returns:
    --------
    list of tuples
        Liste de tuples (train_set, test_set) pour chaque itération
    """
    
    # Convertir les périodes en timedelta
    train_delta = pd.Timedelta(train_period)
    test_delta = pd.Timedelta(test_period)
    period_delta = pd.Timedelta(period)
    
    # S'assurer que la colonne ds est en datetime
    df = df.copy()
    #df['ds'] = pd.to_datetime(df['ds'])
    #df = df.sort_values('ds').reset_index(drop=True)
    
    # Date de fin du dataset
    end_date = df['ds'].max()
    
    train_test_sets = []
    
    for i in range(n_iter):
        # Calculer les dates pour cette itération
        # Décaler vers le passé selon l'itération
        current_end = end_date - (period_delta * i)
        
        # Dates du test_set (les plus récentes)
        test_start = current_end - test_delta
        test_end = current_end
        
        # Dates du train_set (avant le test_set)
        train_start = test_start - train_delta
        train_end = test_start
        
        # Filtrer les données
        train_set = df[(df['ds'] >= train_start) & (df['ds'] < train_end)].copy()
        test_set = df[(df['ds'] >= test_start) & (df['ds'] < test_end)].copy()
        
        # Ajouter seulement si les sets ne sont pas vides
        if len(train_set) > 0 and len(test_set) > 0:
            train_test_sets.append((train_set, test_set))
        else:
            print(f"Itération {i+1}: Pas assez de données pour train_start={train_start}")
            break

    train_test_sets.reverse()
    
    return train_test_sets

# train and predict for each train_test_set
def train_predict(train_test_sets, regressors=[]):

    results = []
    for i, tt_set in enumerate(train_test_sets): 
        print(f"SET {i}") 
        print(f"train from {tt_set[0]['ds'].min()} to {tt_set[0]['ds'].max()}")
        print(f"test from {tt_set[1]['ds'].min()} to {tt_set[1]['ds'].max()}")
        # define train and test set
        train_set = tt_set[0]
        test_set = tt_set[1]
        print('columns',train_set.columns)

        # model
        model = Prophet()
        if len(regressors) > 0:
            for regressor in regressors:
                model.add_regressor(regressor)
        model.fit(train_set)
        fields = ['ds'] + regressors
        train_pred = model.predict(train_set[fields])
        test_pred = model.predict(test_set[fields])

        results.append((train_pred, test_pred))
        
        # Calculate metrics
        train_mae = mean_absolute_error(train_set['y'], train_pred['yhat'])
        test_mae = mean_absolute_error(test_set['y'], test_pred['yhat'])
        train_mape = mean_absolute_percentage_error(train_set['y'], train_pred['yhat'])
        test_mape = mean_absolute_percentage_error(test_set['y'], test_pred['yhat'])
        
        print(f'Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}')
        print(f'Train MAPE: {100*train_mape:.2f}, Test MAPE: {100*test_mape:.2f}')
        print('-' * 50)
    return results
