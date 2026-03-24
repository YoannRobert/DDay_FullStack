import pandas as pd
import streamlit as st
import numpy as np
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(layout="wide")

st.html(
    """
    <style>
    .block-container {
        padding-top: 3rem !important;
    }
    </style>
    """
)


st.header("Application de prédiction de consommation électrique")

st.divider()

page_prediction = st.Page("page_prediction.py", title="Prédiction", icon=":material/model_training:")
page_historic = st.Page("page_historic.py", title="Historique", icon=":material/history:")

pg = st.navigation([page_prediction, page_historic])
pg.run()
