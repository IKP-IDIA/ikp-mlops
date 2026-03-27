import mlflow
from mlflow.tracking import MlflowClient
import os

def promote_latest_to_production(model_name: str):
  local_tracking_uri = "http://10.1.0.150:5000"
  mlflow.set_tracking_uri(local_tracking_uri)
  client = MlflowClient()
  
  try:
    latest_versions = client.get_latest_versions(model_name)
    if not latest_versions:
      print(f"Not found the latest in '{model_name}' ")
      return
    
    latest_v = latest_versions[0].version
    print(f"Found {model_name} version {latest_v} moving to Production...")

    # 2. Move Stage to Production
    client.transition_model_version_stage(
      name=model_name,
      version=latest_v,
      stage="Production",
      archive_existing_versions=True # Keep old version of Production automatic
      )
    
    print(f"✅ Done! Model {model_name} version {latest_v} is now in Production.")
    print(f"🚀 Ready for Serving (e.g., KServe, BentoML, or Fast API)")

  except Exception as e:
    print(f" Error: {e}")

if __name__ == "__main__":
    MODEL_NAME = "FraudDetection_Model" # same name in Stage 04
    promote_latest_to_production(MODEL_NAME)