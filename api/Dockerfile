FROM python:3.10-slim

# --- Dépendances système ---
# build-essential / gcc / g++ : compilation de Prophet (pystan) et pmdarima
# libgomp1                    : OpenMP requis par XGBoost et Prophet
# libpq-dev                   : psycopg2-binary (connexion PostgreSQL / Neon)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Génération de requirements-api.txt depuis environment.yml ---
# PyYAML est nécessaire pour exécuter le script de génération.
# On copie uniquement les fichiers nécessaires à cette étape pour exploiter le cache Docker.
COPY environment.yml requirements.txt pyproject.toml generate_requirements_api.py ./

RUN pip install --no-cache-dir pyyaml \
    && python generate_requirements_api.py

# --- Installation des dépendances Python ---
RUN pip install --no-cache-dir -r requirements-api.txt

# --- Installation du projet en mode éditable ---
# Rend les imports absolus fonctionnels (from etl.etl import ...)
# depuis api/main.py, exactement comme -e . dans environment.yml.
RUN pip install --no-cache-dir -e .

# --- Activation des imports absolus depuis la racine du projet ---
ENV PYTHONPATH=/app

# --- Copie du code source ---
COPY . .

EXPOSE 8000

# main.py se trouve dans le sous-dossier api/
CMD ["python", "api/main.py"]