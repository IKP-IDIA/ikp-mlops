FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    mlflow==2.11.3 \
    kserve==0.11.2 \
    pandas \
    joblib \
    scikit-learn \
    boto3 \
    python-box \
    ensure \
    python-dotenv \
    protobuf==3.20.3

COPY . .

ENV PYTHONPATH="/app/src"

EXPOSE 8080

#CMD ["python", "app.py"]
CMD ["python", "src/fraud_prediction/model_kserve.py"]