import os
from fraud_prediction.constants import *
from fraud_prediction.utils.common import read_yaml, create_directories
from fraud_prediction.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig,
                                                MonitoringConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            onedrive_file_path="None",
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            update_base_model_path=Path(config.update_base_model_path),
            params_num_features=self.params.NUM_FEATURES,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = self.config.data_ingestion.unzip_dir
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            update_base_model_path=Path(prepare_base_model.update_base_model_path),
            training_data=Path(training_data),
            experiment_name=training.experiment_name,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_num_features=params.NUM_FEATURES,
            params_sampling_ratio=params.SAMPLING_RATIO
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_values = self.config.evaluation 
        params = self.params

        create_directories([eval_values.root_dir])
        
        eval_config = EvaluationConfig(
            path_of_model=eval_values.path_of_model,
            all_params=params,
            mlflow_uri=eval_values.mlflow_uri,
            training_data=eval_values.training_data,
            experiment_name=eval_values.experiment_name,
            registered_model_name=eval_values.registered_model_name,
            # root_dir=eval_values.root_dir,
            # path_of_model=Path(self.config.training.trained_model_path),
            # training_data=Path(self.config.data_ingestion.unzip_dir),
            # mlflow_uri=self.config.mlflow_uri,
            # all_params=self.params,
            # experiment_name=self.config.training.experiment_name,
            params_num_features=self.params.NUM_FEATURES,
            params_batch_size=self.params.BATCH_SIZE,
            # registered_model_name=eval_values.registered_model_name
        )
        return eval_config

    def get_monitoring_config(self) -> MonitoringConfig:
        # ดึงส่วนของ monitoring จาก yaml
        config = self.config.model_monitoring 
        
        # สร้าง directory สำหรับเก็บผล monitoring
        create_directories([Path(config.root_dir)])
    
        monitoring_config = MonitoringConfig(
            root_dir=Path(config.root_dir),
            # ใช้ unzip_dir จาก data_ingestion เพื่อไปหา test.csv
            training_data=Path(self.config.data_ingestion.unzip_dir), 
            current_data_path=Path(config.current_data_path),
            # ดึง mlflow_uri จากจุดที่กำหนดไว้ใน config.yaml (ตรวจสอบให้ตรงกับที่ Evaluation ใช้)
            mlflow_uri=self.config.evaluation.mlflow_uri, 
            experiment_name=self.config.evaluation.experiment_name,
            drift_report_path=Path(config.drift_report_path)
        )
        
        return monitoring_config