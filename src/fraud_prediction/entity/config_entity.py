from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    onedrive_file_path: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    update_base_model_path: Path
    params_num_features: int
    params_learning_rate: float
    params_include_top: bool
    params_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    update_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_num_features: int
    params_sampling_ratio: int 
    experiment_name: str


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_num_features: int
    params_batch_size: int
    experiment_name: str
    registered_model_name: str
    
@dataclass(frozen=True)
class MonitoringConfig:
    root_dir: Path
    training_data: Path # สำหรับดึง test.csv (Reference)
    current_data_path: Path # สำหรับดึงข้อมูลใหม่มาเช็ค (Current)
    mlflow_uri: str
    experiment_name: str
    drift_report_path: Path # ที่เก็บไฟล์ .html ของ Evidently