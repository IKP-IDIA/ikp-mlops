from fraud_prediction.config.configuration import ConfigurationManager
from fraud_prediction.components.model_trainer import Training
from fraud_prediction import logger
import mlflow 
from datetime import datetime


STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self,experiment_name):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        
        # Run main name (Parent)
        #run_name = f"Fraud_Training_Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="Stage_03_Model_Training", nested=True) as run:
          logger.info(f"Active Parent Run ID : {run.info.run_id}")
          
          # Parent
          mlflow.log_param("stage", "Model Training")
        
          training = Training(config=training_config)
          training.get_base_model()
          training.prepare_data()
          training.train(experiment_name=experiment_name) #create child run in side this step


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        