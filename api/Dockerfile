FROM continuumio/miniconda3

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

# --- Création de l'environnement Conda ---
# On copie d'abord les fichiers de config pour exploiter le cache Docker :
# tant que ces fichiers ne changent pas, cette couche n'est pas reconstruite.
COPY environment.yml requirements.txt pyproject.toml ./

RUN conda env create -f environment.yml \
    && conda clean -afy

# --- Activation de l'environnement Conda dans tous les RUN / CMD suivants ---
SHELL ["/bin/bash", "--login", "-c"]
ENV PATH="/opt/conda/envs/dday/bin:$PATH"

# --- Copie du code source ---
COPY . .

EXPOSE 8000

# main.py se trouve dans le sous-dossier api/
# On utilise le Python de l'environnement Conda "dday" (via PATH ci-dessus)
CMD ["python", "api/main.py"]