FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD mlflow server -p $PORT \
--host 0.0.0.0 \
--allowed-hosts ${HF_DEMODAY_SPACE_URL} \
--cors-allowed-origins "https://${HF_DEMODAY_SPACE_URL}" \
--backend-store-uri "postgresql://${NEON_USERNAME}:${NEON_PASSWORD}@${NEON_HOST}/neondb?sslmode=require" \
--default-artifact-root $ARTIFACT_STORE_URI
