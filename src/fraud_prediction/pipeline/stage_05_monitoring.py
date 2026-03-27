from fraud_prediction.config.configuration import ConfigurationManager
from fraud_prediction.components.model_monitoring import ModelMonitoring
from fraud_prediction import logger

STAGE_NAME = "Model Monitoring Stage"

class ModelMonitoringPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        config = ConfigurationManager
        monitoring_config = config.get_monitoring_config()
        monitoring = ModelMonitoring(config=monitoring_config)
        monitoring.run_monitoring()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelMonitoringPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise elo