import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import timedelta
from dotenv import load_dotenv
import utils


load_dotenv()

st.set_page_config(layout="wide")


df = utils.load_recent_dataset()

df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
df['end_date'] = pd.to_datetime(df['end_date'], utc=True)
df['start_date_fr'] = df['start_date'].dt.tz_convert('Europe/Paris')
df['end_date_fr'] = df['end_date'].dt.tz_convert('Europe/Paris')

df_pred = utils.load_prediction_dataset()
df_pred['ds_fr'] = pd.to_datetime(df_pred['ds'], utc=True).dt.tz_convert('Europe/Paris')

df_recent = df.copy()
df_recent.dropna(inplace=True)

# Last day with observed data (Paris calendar, consistent with the filters below)
data_end_date = df_recent['start_date_fr'].max().date()
last_obs_end = df_recent['end_date_fr'].max()

# Persisted period selection (managed by st.pills via its key="select_period")
if "select_period" not in st.session_state:
    st.session_state.select_period = 'J-1'

# Derive the display window at every rerun, so a fresh predict() that updates
# data_end_date automatically realigns the window without needing a manual refresh.
n_days = {'J-1': 0, 'J-3': 2, 'J-7': 6}.get(st.session_state.select_period, 0)
pred_start_date = data_end_date - timedelta(days=n_days)
pred_end_date = data_end_date

mask_start_date = (df['start_date_fr'].dt.date >= pred_start_date)
mask_end_date = (df['start_date_fr'].dt.date <= pred_end_date)
df2 = df[mask_start_date & mask_end_date]

# Predictions are kept from pred_start_date onward (Paris calendar), matching
# the start of the consumption curve. This makes the green curve overlap the
# blue one on the past period, enabling visual retrospective comparison of
# past predictions vs actual consumption. The forecast tail beyond
# last_obs_end is naturally preserved.
df_pred2 = df_pred[df_pred['ds_fr'].dt.date >= pred_start_date]


# Metrics for the observed period (blue curve)
mean_power = df2['consumption_MW'].mean()
min_power = df2['consumption_MW'].min()
max_power = df2['consumption_MW'].max()

# Metrics for the prediction restricted to the same observed window, so the
# delta against the consumption metrics compares two values on the same time range.
df_pred_for_metrics = df_pred2[df_pred2['ds_fr'] <= last_obs_end]
mean_power_pred = df_pred_for_metrics['yhat'].mean()
min_power_pred = df_pred_for_metrics['yhat'].min()
max_power_pred = df_pred_for_metrics['yhat'].max()

delta_mean_power = mean_power_pred - mean_power
delta_min_power = min_power_pred - min_power
delta_max_power = max_power_pred - max_power


st.html("""
<style>
    div.st-key-mean_metrics, div.st-key-min_metrics, div.st-key-max_metrics {
        min-height: 135px !important;
    }
    div.st-key-min_metrics [data-testid="stMetricValue"],
    div.st-key-max_metrics [data-testid="stMetricValue"] {
    font-size: 24px !important;
    }
</style>
""")


###############################################
# Subtitle
if pred_start_date == pred_end_date:
    st.subheader(f"Puissance électrique consommée le {pred_start_date.strftime('%d/%m/%Y')}")
else:
    st.subheader(f"Puissance électrique consommée entre le {pred_start_date.strftime('%d/%m/%Y')} et le {pred_end_date.strftime('%d/%m/%Y')}")


###############################################
# Metrics power

col1_metrics, col2_metrics, col3_metrics = st.columns((4, 2, 2))

with col1_metrics:
    with st.container(border=True, key="mean_metrics"):
        col11_prod, col12_pred = st.columns(2)
        with col11_prod:
            st.metric("Puissance moyenne réelle", f"{mean_power:.2f} MW")
        with col12_pred:
            st.metric("Puissance moyenne prédite", f"{mean_power_pred:.2f} MW", f"{delta_mean_power:.2f} MW")

with col2_metrics:
    with st.container(border=True, key="min_metrics"):
        col21_prod, col22_pred = st.columns(2)
        with col21_prod:
            st.metric("Puissance minimale réelle", f"{min_power:.2f} MW")
        with col22_pred:
            st.metric("Puissance minimale prédite", f"{min_power_pred:.2f} MW", f"{delta_min_power:.2f} MW")

with col3_metrics:
    with st.container(border=True, key="max_metrics"):
        col31_prod, col32_pred = st.columns(2)
        with col31_prod:
            st.metric("Puissance maximale réelle", f"{max_power:.2f} MW")
        with col32_pred:
            st.metric("Puissance maximale prédite", f"{max_power_pred:.2f} MW", f"{delta_max_power:.2f} MW")

##############################################


###############################################
# Filtre période
option_days_selection = ['J-1', 'J-3', 'J-7']

col1_btn, col2_btn = st.columns(2, vertical_alignment="bottom")
with col1_btn:
    st.pills(
        "Sélection de la période",
        key="select_period",
        options=option_days_selection,
        selection_mode="single",
        label_visibility="collapsed",
    )
with col2_btn:
    # Clear cache
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("", icon=":material/refresh:", help="Actualiser les données"):
            st.cache_data.clear()


################################################
# Chart
fig = px.line(df2, x='end_date_fr', y='consumption_MW', range_y=[20000, None],
              title="Puissance électrique consommée et prédictions en MW",
              labels={'end_date_fr': '', 'consumption_MW': 'Consommation (MW)'})
fig.add_trace(px.line(df_pred2, x='ds_fr', y='yhat').data[0])

fig.data[0].update({'name': 'Consommation'})
fig.data[1].update({'line': dict(dash='dash', color="green"), 'name': 'Prédiction'})
fig.update_traces(showlegend=True)

with st.container(border=1):
    st.plotly_chart(fig)
