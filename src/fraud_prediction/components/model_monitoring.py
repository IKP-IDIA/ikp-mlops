import pandas as pd
import mlflow
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from fraud_prediction.entity.config_entity import EvaluationConfig
from evidently.pipeline.column_mapping import ColumnMapping
#from evidently.model_profile import Profile
#from evidently.pipeline.column_mapping import ColumnMapping

class ModelMonitoring:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def run_drift_analysis(self, current_data_path: str, run_name="05_Drift_Monitoring"):
        """
        Compare Training data (Reference) with New Data (Current)
        """
        
        # Load  Reference Data (Ex. Lastest training data)
        # Normally pull form artifacts/data_ingestion/test.csv
        ref_df = pd.read_csv(os.path.join(self.config.training_data, "test.csv"))
        
        # Download current data ( Data from Poduction or Inference Log)
        cur_df = pd.read_csv(current_data_path)
        
        column_mapping = ColumnMapping()
        column_mapping.target = 'isFraud'
        column_mapping.numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                                            'oldbalanceDest', 'newbalanceDest', 
                                            'diff_new_old_balance', 'diff_new_old_destiny']
        
        
        # Build Evidently Report
        # Check both Data Drift and Data Quanlity
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DataQualityPreset()
        ])
        
        report.run(reference_data=ref_df, current_data=cur_df)
        report_html_path = str(self.config.drift_report_path)
        report.save_html(str(self.config.drift_report_path))
        
        # Save results to MLflow
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        with mlflow.start_run(run_name=f"Monitoring_Drift_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"):
            # เก็บไฟล์ HTML เป็น Artifact ใน MLflow
            mlflow.log_artifact(report_html_path)
            
            # ดึงค่า Metrics สำคัญมาบันทึกเป็นเลข (Log Metric)
            metrics_dict = report.as_dict()
            # โครงสร้าง Dict ของ v0.4 อาจลึกหน่อย ต้องดึงให้ถูกจุด
            # โดยปกติ DataDriftPreset จะอยู่ที่ตำแหน่ง 0
            drift_share = metrics_dict['metrics'][0]['result']['drift_share']
            
            mlflow.log_metric("data_drift_share", drift_share)
            print(f"Drift Analysis Done! Share: {drift_share:.2%}")
            
            # 5. Logic สำหรับ Auto-Retrain
            if drift_share > 0.40:
                print("⚠️ Warning: High Drift detected! Recommendation: Retrain Model.")
                return True
            return False
        