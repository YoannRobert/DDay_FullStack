import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events
import datetime

st.set_page_config(layout="wide")

@st.cache_data
def load_dataset():
    df = pd.read_csv('../../data/csv/power_consumption_2000_2025_hourly.csv')
    df['start_date'] = pd.to_datetime(df['start_date'], utc=True).dt.tz_convert('Europe/Paris')
    df['end_date'] = pd.to_datetime(df['end_date'], utc=True).dt.tz_convert('Europe/Paris')
    return df


df = load_dataset()


st.header("Application de prédiction de consommation électrique")

st.divider()


# Session State
default_start_date = datetime.date(2020, 1, 1)
default_end_date = datetime.date(2025, 12, 31)

# start_datetime of data
if "start_date" not in st.session_state:
    st.session_state.start_date = default_start_date

# end_datetime of data
if "end_date" not in st.session_state:
    st.session_state.end_date = default_end_date

def update_range_year():
    print(st.session_state.range_year)
    st.session_state.start_date = datetime.date(st.session_state.range_year[0], 1, 1)
    st.session_state.end_date = datetime.date(st.session_state.range_year[1], 12, 31)
    st.session_state.inp_start_date = st.session_state.start_date
    st.session_state.inp_end_date = st.session_state.end_date

def update_start_datetime():
    print(st.session_state.inp_start_date)
    st.session_state.start_date = st.session_state.inp_start_date
    st.session_state.range_year = (st.session_state.start_date.year, st.session_state.end_date.year)

def update_end_datetime():
    print(st.session_state.inp_end_date)
    st.session_state.end_date = st.session_state.inp_end_date
    st.session_state.range_year = (st.session_state.start_date.year, st.session_state.end_date.year)


mask_start_date = (df['start_date'].dt.date >= st.session_state.start_date) 
mask_end_date = (df['end_date'].dt.date <= st.session_state.end_date)
df2 = df[mask_start_date & mask_end_date]

mean_power = df2['value'].mean()
min_power = df2['value'].min()
max_power = df2['value'].max()

###############################################
# Subtitle

st.subheader(f"Puissance électrique consommée entre le {st.session_state.start_date} et le {st.session_state.end_date}")


###############################################
# Metrics power

col1_stats, col2_stats, col3_stats = st.columns(3)

with col1_stats:
    st.metric("Puissance moyenne", f"{mean_power:.2f} MW", border=True)
with col2_stats:
    st.metric("Puissance minimale", f"{min_power:.2f} MW", border=True)
with col3_stats:
    st.metric("Puissance maximale", f"{max_power:.2f} MW", border=True)

###############################################
# Filters

# Year slider
start_year, end_year = st.slider( label="Année", key="range_year",
    min_value=2000,
    max_value=2025,
    value=(2020,2025),
    on_change=update_range_year
)

# Month slider
"""start_month, end_month = st.slider( label="Mois", key="range_month",
    min_value='2000-01',
    max_value='2025-01',
    value=(1,12),
    on_change=update_range_month
)"""

col1_date, col2_date = st.columns(2)
# Date calendar selector
with col1_date:
    start_time = st.date_input( label="Date de début", key="inp_start_date",
        value=default_start_date,
        min_value=datetime.date(2000, 1, 1),
        max_value=datetime.date(2025, 12, 31),
        on_change=update_start_datetime
    )
with col2_date:
    end_time = st.date_input( label="Date de fin", key="inp_end_date",
        value=default_end_date,
        min_value=datetime.date(2000, 1, 1),
        max_value=datetime.date(2025, 12, 31),
        on_change=update_end_datetime
    )

################################################
# Chart

fig2 = px.line(df2, x='start_date', y='value', render_mode="svg",
    title=f"Production d'électricité entre {st.session_state.start_date} et {st.session_state.end_date} en MW")
fig2.update_xaxes(rangeslider_visible=True)

with st.container(border=1):
    st.plotly_chart(fig2)
