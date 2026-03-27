import pandas as pd
import mlflow
import os
import glob
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from fraud_prediction.entity.config_entity import EvaluationConfig
from evidently.pipeline.column_mapping import ColumnMapping
#from evidently.model_profile import Profile
#from evidently.pipeline.column_mapping import ColumnMapping

class ModelMonitoring:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def run_drift_analysis(self, current_data_path: str):
        """
        Compare Training data (Reference) with New Data (Current)
        """
        
        # Load  Reference Data (Ex. Lastest training data)
        # Normally .ull form artifacts/data_ingestion/test.csv
        data_dir = self.config.training_data
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        
        if not csv_files:
            raise FileNotFoundError(f"Not found CSV in {data_dir}")
        
        ref_df = pd.read_csv(csv_files[0])
        # Download current data ( Data from Poduction or Inference Log)
        cur_df = pd.read_csv(current_data_path)
        
        for df in [ref_df, cur_df]:
            df['diff_new_old_balance'] = df['newbalanceOrig'] - df['oldbalanceOrg']
            df['diff_new_old_destiny'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        SAMPLE_SIZE = 10000
        ref_df = ref_df.sample(n=min(SAMPLE_SIZE, len(ref_df)), random_state=42)
        cur_df = cur_df.sample(n=min(SAMPLE_SIZE, len(cur_df)), random_state=42)
        #print(f"📉 Optimized: Processing {len(cur_df)} rows for drift analysis.")
        
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
        
        # Build & Run Evidently Report
        #report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
        
        metrics_dict = report.as_dict()
        drift_results = metrics_dict.get("metrics",[])
        
        drift_metrics = {}
        for m in drift_results:
            result = m.get("result",{})
            #DataDriftPreset
            if "drift_share" in result:
                drift_metrics["drift_share"] = result["drift_share"]
                drift_metrics["number_of_drifted_columns"] = result.get("number_of_drifted_columns",0)
                drift_metrics["number_of_columns"]= result.get("number_of_columns",0)
            #TargetDriftPreset (if have drift_detected)
            if "drift_detected" in result:
                drift_metrics["targer_drift_detected"]=int(result["drift_detected"])
        
        # Save HTML locally
        report_html_path = "artifacts/monitoring/drift_report.html"
        os.makedirs(os.path.dirname(report_html_path), exist_ok=True)
        report.save_html(report_html_path)
        
        # Save results to MLflow
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        # ใช้ nested=True เพื่อให้ไปอยู่ภายใต้วงจร Pipeline หลัก
        with mlflow.start_run(run_name=f"Monitoring_Drift_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}", nested=True):
            #mlflow.log_artifact(local_path=report_html_path, artifact_path="drift_reports")
            mlflow.log_metrics(drift_metrics)
            
            report_html_path = "artifacts/monitoring/drift_report.html"
            os.makedirs(os.path.dirname(report_html_path), exist_ok=True)
            report.save_html(report_html_path)
            mlflow.log_artifact(report_html_path, artifact_path="drift_reports")
            
            current_drift_share = drift_metrics.get("data_drift",0.0)
            print(f" Drift Analysis Done! Share: {current_drift_share:.2%}")
            
            if current_drift_share > 0.40:
                print("⚠️ Warning: High Drift detected! Recommendation: Retrain Model.")
                return True
            return False
        #return drift_share
        