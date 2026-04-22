import kserve
from typing import Dict
import pandas as pd
from fraud_prediction.utils.common import preprocess_fraud_data # ใช้ฟังก์ชันที่เราแยกไว้
import mlflow.pyfunc
from fraud_prediction.utils.common import preprocess_fraud_data, load_json
from pathlib import Path
import sys
import os
sys.path.append(os.getcwd())

class FraudModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False

    def load(self):
        # ตั้งค่า Environment ให้ Botocore มองเห็น
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://10.1.0.250:9000"
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "default_key")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY", "default_secret")
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
        
        model_uri = "s3://mlflow/1/8c7c175a3c764f4aaf48677c5666d25a/artifacts/model"
        
        print(f"--- Downloading model from: {model_uri} ---")
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.ready = True
        print("--- Model Ready! ---")

    def predict(self, payload , headers: dict = None):
        # รับข้อมูล JSON
        instances = payload["instances"]
        df_raw = pd.DataFrame(instances)
        
        # ใช้ฟังก์ชันที่เราเพิ่งเขียนใน common.py
        df_processed = preprocess_fraud_data(df_raw)
        
        # ทำนายผล
        prediction = self.model.predict(df_processed)
        return {"predictions": prediction.round(6).tolist()}

if __name__ == "__main__":
    model = FraudModel("fraud-detection")
    model.load()
    kserve.ModelServer(http_port=8080, grpc_port=8085).start([model])