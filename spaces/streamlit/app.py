import streamlit as st
import requests
from dotenv import load_dotenv
from time import sleep
import os
import utils

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

with st.sidebar:
    if st.button("Lancer une prédiction", use_container_width=True):
        with st.spinner("Chargement en cours..."):
            try:
                sleep(2)
                response = requests.get(f"{PREDICTION_API_URL}/predict")
                if response.status_code == 200:
                    # Targeted invalidation: only the two datasets impacted by
                    # the prediction step. The 2021-2025 historical dataset
                    # stays cached.
                    utils.load_recent_dataset.clear()
                    utils.load_prediction_dataset.clear()
                    st.sidebar.success("Prédiction effectuée avec succès !")
                    st.rerun()
                else:
                    st.sidebar.error(f"Erreur : {response.status_code}")
            except Exception as e:
                st.sidebar.error(f"Erreur : {e}")

pg = st.navigation([page_prediction, page_historic])
pg.run()