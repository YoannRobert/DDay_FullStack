import pandas as pd
import streamlit as st
import numpy as np
import datetime
import requests
from dotenv import load_dotenv
from time import sleep
import os

load_dotenv()

PREDICTION_API_URL = os.getenv("PREDICTION_API_URL")

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

# Bouton dans la sidebar, sous la navigation

with st.sidebar:
    if st.button("Lancer une prédiction", use_container_width=True):
        with st.spinner("Chargement en cours..."):
            try:
                sleep(2)
                response = requests.get(f"{PREDICTION_API_URL}/predict")  # Remplacez par votre URL
                if response.status_code == 200:
                    st.cache_data.clear()
                    st.sidebar.success("Prédiction effectuée avec succès !")
                    st.rerun()
                else:
                    st.sidebar.error(f"Erreur : {response.status_code}")
            except Exception as e:
                st.sidebar.error(f"Erreur : {e}")


pg = st.navigation([page_prediction, page_historic])
pg.run()
