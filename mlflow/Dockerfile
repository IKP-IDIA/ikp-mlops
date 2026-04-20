FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow[extras]==2.19.0 \
    psycopg2-binary \
    boto3 \
    cryptography

EXPOSE 5000