FROM python:3.10-slim

RUN pip install mlflow[auth] boto3 pymysql
# ติดตั้ง dependencies สำหรับระบบ
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง MLflow และ library ที่ต้องใช้คุยกับ S3 (MinIO) และ Postgres
RUN pip install \
    mlflow[extras] \
    psycopg2-binary \
    boto3 \
    cryptography

EXPOSE 5000