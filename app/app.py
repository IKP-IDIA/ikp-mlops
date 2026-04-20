import os
import pandas as pd
import joblib
import mlflow.pyfunc
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from typing import Optional

load_dotenv()  # โหลด .env เหมือนกับ main.py

# ✅ set credentials ก่อน load model
os.environ["MLFLOW_TRACKING_URI"]       = os.getenv("MLFLOW_TRACKING_URI", "http://10.1.0.150:5000")
os.environ["MLFLOW_S3_ENDPOINT_URL"]    = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://10.1.0.250:9000")
os.environ["AWS_ACCESS_KEY_ID"]         = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"]     = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_IGNORE_TLS"]      = "true"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: Optional[int] = None 
    isFlaggedFraud: Optional[int] = None 

app = FastAPI(title="Fraud Detection Service (MLOps)")

model = None
scaler = None

try:
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='FraudDetection_Model'")
    
    if not versions:
        raise Exception("No model versions found")
    
    # เรียงตาม version ล่าสุด
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    RUN_ID = latest.run_id
    MODEL_URI = f"runs:/{RUN_ID}/model"
    print(f"Found model version {latest.version} from run {RUN_ID}")

    # load model
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"Model loaded from {MODEL_URI}")
    
    os.makedirs("/tmp/mlflow_artifacts", exist_ok=True)
    experiment   = client.get_experiment_by_name("Fraud_Detection_v2")
    training_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Model_Training_Fit'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not training_runs:
        raise Exception("Model_Training_Fit run not found")

    SCALER_RUN_ID     = training_runs[0].info.run_id
    os.makedirs("/tmp/mlflow_artifacts", exist_ok=True)
    scaler_local_path = client.download_artifacts(SCALER_RUN_ID, "scaler.pkl", "/tmp/mlflow_artifacts")
    scaler = joblib.load(scaler_local_path)
    print(f"Scaler loaded from Training run {SCALER_RUN_ID}")
    
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    
FEATURE_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "diff_new_old_balance", "diff_new_old_destiny",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]

@app.get("/health")
async def health():
    return {
        "status":       "ok" if model and scaler else "degraded",
        "model_loaded"  : model is not None,
        "scaler_loaded" : scaler is not None,
    }

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not defined or failed to load")

    try:
        diff_new_old_balance = transaction.newbalanceOrig - transaction.oldbalanceOrg
        diff_new_old_destiny = transaction.newbalanceDest - transaction.oldbalanceDest

        types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        type_dummies = {f"type_{t}": (1.0 if transaction.type == t else 0.0) for t in types}

        feature_dict = {
            "step":                  float(transaction.step),
            "amount":                transaction.amount,
            "oldbalanceOrg":         transaction.oldbalanceOrg,
            "newbalanceOrig":        transaction.newbalanceOrig,
            "oldbalanceDest":        transaction.oldbalanceDest,
            "newbalanceDest":        transaction.newbalanceDest,
            "diff_new_old_balance":  diff_new_old_balance,
            "diff_new_old_destiny":  diff_new_old_destiny,
            **type_dummies,
        }

        # Build DataFrame for column order same training
        input_df   = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
        
        # apply scaler before predict (same train/eval)
        input_scaled = scaler.transform(input_df)
        input_scaled = pd.DataFrame(input_scaled, columns=FEATURE_COLUMNS).astype('float32')
        
        prediction = model.predict(input_scaled)
        score      = float(prediction[0])

        return {
            "is_fraud":  bool(score > 0.5),
            "score":     score,
            "model_uri": MODEL_URI,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)