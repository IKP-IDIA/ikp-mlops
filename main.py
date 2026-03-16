import os
import mlflow
from fraud_prediction import logger 
from fraud_prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fraud_prediction.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from fraud_prediction.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from fraud_prediction.pipeline.stage_04_model_evaluation import EvaluationPipeline
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

print(f"Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")

EXPERIMENT_NAME = "Fraud_Detection_v1"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=f"Full_Pipeline_{datetime.now().strftime('%Y%m%d')}"):
  STAGE_NAME = "Data Ingestion stage"
  
  try: 
      logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
      obj = DataIngestionTrainingPipeline()
      obj.main() 
      logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  except Exception as e: 
      logger.exception(e) 
      raise e 
  
  STAGE_NAME = "Prepare base model"
  try:
      logger.info(f"***************") 
      logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
      prepare_base_model = PrepareBaseModelTrainingPipeline() 
      prepare_base_model.main() 
      logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  except Exception as e:
      logger.exception(e) 
      raise e
  
  STAGE_NAME = "Model Training"
  try:
      logger.info(f"*******************")
      logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
      model_trainer = ModelTrainingPipeline()
      model_trainer.main(experiment_name=EXPERIMENT_NAME)
      logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  except Exception as e:
      logger.exception(e)
      raise e
  
  STAGE_NAME = "Evaluation stage"
  try:
     logger.info(f"*******************")
     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
     model_evalution = EvaluationPipeline()
     model_evalution.main(experiment_name=EXPERIMENT_NAME)
     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
  
  except Exception as e:
          logger.exception(e)
          raise e