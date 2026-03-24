import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import datetime
from dotenv import load_dotenv
import utils
import os


load_dotenv()

st.set_page_config(layout="wide")

@st.cache_data
def load_recent_dataset():
    return utils.load_dataset_s3("dataset/training_dataset.csv")

def load_prediction_dataset():
    return utils.load_dataset_s3("dataset/predictions.csv")


df = load_recent_dataset()

df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
df['end_date'] = pd.to_datetime(df['end_date'], utc=True)
df['start_date_fr'] = df['start_date'].dt.tz_convert('Europe/Paris')
df['end_date_fr'] = df['end_date'].dt.tz_convert('Europe/Paris')

df_pred = load_prediction_dataset()
df_pred['ds_fr'] = pd.to_datetime(df_pred['ds'], utc=True).dt.tz_convert('Europe/Paris')


df_recent = df.copy()
df_recent.dropna(inplace=True)

# Last date with data
data_end_date = df_recent['end_date'].max().date()


# Session State
default_start_date = data_end_date
default_end_date = data_end_date

# start_datetime of data
if "pred_start_date" not in st.session_state:
    st.session_state.pred_start_date = default_start_date

# end_datetime of data
if "pred_end_date" not in st.session_state:
    st.session_state.pred_end_date = default_end_date

mask_start_date = (df['start_date_fr'].dt.date >= st.session_state.pred_start_date) 
mask_end_date = (df['end_date_fr'].dt.date <= st.session_state.pred_end_date)
df2 = df[mask_start_date & mask_end_date]

# Calculate metrics for the selected period
mean_power = df2['consumption_MW'].mean()
min_power = df2['consumption_MW'].min()
max_power = df2['consumption_MW'].max()

# calculate metrics for the previous period
previous_day = st.session_state.pred_start_date - datetime.timedelta(days=1)
mask_previous_start = (df['start_date_fr'].dt.date >= previous_day) 
mask_previous_end = (df['end_date_fr'].dt.date <= previous_day)
df_previous = df[mask_previous_start & mask_previous_end]
mean_power_previous = df_previous['consumption_MW'].mean()
min_power_previous = df_previous['consumption_MW'].min()
max_power_previous = df_previous['consumption_MW'].max()

delta_mean_power = mean_power - mean_power_previous
delta_min_power = min_power - min_power_previous
delta_max_power = max_power - max_power_previous



###############################################
# Subtitle
if st.session_state.pred_start_date == st.session_state.pred_end_date:
    st.subheader(f"Puissance électrique consommée le {st.session_state.pred_start_date.strftime('%d/%m/%Y')}")
else:
    st.subheader(f"Puissance électrique consommée entre le {st.session_state.pred_start_date.strftime('%d/%m/%Y')} et le {st.session_state.pred_end_date.strftime('%d/%m/%Y')}")


###############################################
# Metrics power

col1_stats, col2_stats, col3_stats = st.columns(3)

with col1_stats:
    st.metric("Puissance moyenne", f"{mean_power:.2f} MW", f"{delta_mean_power:.2f} MW", border=True)
with col2_stats:
    st.metric("Puissance minimale", f"{min_power:.2f} MW", f"{delta_min_power:.2f} MW", border=True)
with col3_stats:
    st.metric("Puissance maximale", f"{max_power:.2f} MW", f"{delta_max_power:.2f} MW", border=True)


################################################
# Chart

fig = px.line(df2, x='end_date_fr', y='consumption_MW',
    title=f"Puissance électrique consommée et prédictions en MW",
    labels={'end_date_fr': '', 'consumption_MW': 'Consommation (MW)'})
fig.add_trace( px.scatter(df_pred, x='ds_fr', y='yhat').data[0] )
fig.data[1].marker = dict(color='green', size=5)

with st.container(border=1):
    st.plotly_chart(fig)
