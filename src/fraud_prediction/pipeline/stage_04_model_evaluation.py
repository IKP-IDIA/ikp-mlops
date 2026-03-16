from fraud_prediction.config.configuration import ConfigurationManager
from fraud_prediction.components.model_evaluation_mlflow import Evaluation
from fraud_prediction import logger
import mlflow


STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self, experiment_name):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        
        with mlflow.start_run(run_name="Stage_04_Model_Evaluation", nested=True):
            evaluation.log_into_mlflow(experiment_name=experiment_name)

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
            