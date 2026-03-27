import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import mlflow
from fraud_prediction import logger 
from fraud_prediction.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fraud_prediction.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from fraud_prediction.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from fraud_prediction.pipeline.stage_04_model_evaluation import EvaluationPipeline
from fraud_prediction.components.model_monitoring import ModelMonitoring
from fraud_prediction.config.configuration import ConfigurationManager

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

now = datetime.now().strftime("%Y%m%d_%H%M")
PARENT_RUN_NAME = f"Fraud_Pipeline_Run_{now}"
EXPERIMENT_NAME = "Fraud_Detection_v2"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. เริ่มต้น Parent Run
with mlflow.start_run(run_name=PARENT_RUN_NAME):
    
    # --- 1. Data Ingestion ---
    STAGE_NAME = "01_Data_Ingestion"
    try: 
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        with mlflow.start_run(run_name=STAGE_NAME, nested=True):
            obj = DataIngestionTrainingPipeline()
            obj.main() 
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e: 
        logger.exception(e) 
        raise e 

    # --- 2. Prepare Base Model ---
    STAGE_NAME = "02_Prepare_Base_Model"
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        with mlflow.start_run(run_name=STAGE_NAME, nested=True):
            prepare_base_model = PrepareBaseModelTrainingPipeline() 
            prepare_base_model.main() 
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e) 
        raise e

    # --- 3. Model Training ---
    STAGE_NAME = "03_Model_Training"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        with mlflow.start_run(run_name=STAGE_NAME, nested=True):
            model_trainer_pipeline = ModelTrainingPipeline()
            model_trainer_pipeline.main(experiment_name=EXPERIMENT_NAME)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    # --- 4. Evaluation Stage ---
    STAGE_NAME = "04_Model_Evaluation"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #with mlflow.start_run(run_name=STAGE_NAME, nested=True):
        model_evaluation_pipeline = EvaluationPipeline()
        # ส่ง run_name เข้าไปเพื่อให้ข้างในใช้ชื่อเดียวกัน
        model_evaluation_pipeline.main(experiment_name=EXPERIMENT_NAME)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    # --- 5. Model Monitoring (Evidently AI) ---
    STAGE_NAME = "05_Model_Monitoring"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #with mlflow.start_run(run_name=STAGE_NAME, nested=True):
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_evaluation_config()
        
        monitoring = ModelMonitoring(config=eval_config)
        # ใส่ Path ข้อมูลที่ต้องการเช็ค Drift
        actual_data_path = "artifacts/data_ingestion/fraud_0.1origbase.csv"
        
        need_retrain = monitoring.run_drift_analysis(
            current_data_path=actual_data_path,
            #run_name=STAGE_NAME # ส่งชื่อ Stage เข้าไปใช้
        )
        
        if need_retrain:
            logger.warning("Drift detected! Recommendation: Trigger Retraining.")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        pass