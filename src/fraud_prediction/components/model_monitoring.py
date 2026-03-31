import pandas as pd
import mlflow
import os
import glob
import json 
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.pipeline.column_mapping import ColumnMapping
from fraud_prediction.entity.config_entity import EvaluationConfig
#from evidently.model_profile import Profile
#from evidently.pipeline.column_mapping import ColumnMapping

class ModelMonitoring:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def run_drift_analysis(self, current_data_path: str):
        """
        Compare Training data (Reference) with New Data (Current)
        """
        
        # ----- load data -----
        data_dir = self.config.training_data
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"Not found CSV in {data_dir}")
        
        ref_df = pd.read_csv(csv_files[0])
        cur_df = pd.read_csv(current_data_path)
        
        for df in [ref_df, cur_df]:
            df['diff_new_old_balance'] = df['newbalanceOrig'] - df['oldbalanceOrg']
            df['diff_new_old_destiny'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        SAMPLE_SIZE = 10000
        ref_df = ref_df.sample(n=min(SAMPLE_SIZE, len(ref_df)), random_state=42)
        cur_df = cur_df.sample(n=min(SAMPLE_SIZE, len(cur_df)), random_state=42)
        #print(f"📉 Optimized: Processing {len(cur_df)} rows for drift analysis.")
        
        # ---- Column mapping ----
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
        
        # ----- Extract metrics -----
        metrics_dict = report.as_dict()
        drift_results = metrics_dict.get("metrics",[])
        
        drift_metrics    ={}
        per_column_drift ={}
        
        for m in drift_results:
            result = m.get("result", {})
            metric_name = m.get("metric", "")

            if metric_name == "DatasetDriftMetric":
                print(f"Full DatasetDriftMetric result: {result}")

            if metric_name == "DatasetDriftMetric":
                drift_metrics["drift_share"]            = result.get("share_of_drifted_columns", 0.0)  # สัดส่วน column ที่ drift
                drift_metrics["drifted_columns"]        = result.get("number_of_drifted_columns", 0)
                drift_metrics["total_columns"]          = result.get("number_of_columns", 0)
                drift_metrics["dataset_drift_detected"] = int(result.get("dataset_drift", False))

            if metric_name == "DataDriftTable":
                # เก็บ drift score ต่อ column for build plot 
                drift_by_cols = result.get("drift_by_columns", {})
                for col_name, col_result in drift_by_cols.items():
                    per_column_drift[col_name]={
                        "drift_score":      float(col_result.get("drift_score",0.0)),
                        "drift_detected":   bool(col_result.get("drift_detected",False)),
                        "stattest":         col_result.get("stattest_name",""),
                        
                    }
            if metric_name == "ColumnDriftMetric":
                if result.get("column_name") == "isFraud":
                    drift_metrics["target_drift_detected"] = int(result.get("drift_detected", False))
                    drift_metrics["target_drift_score"] = float(result.get("drift_score", 0.0))

                    
        current_drift_share = drift_metrics.get("drift_share", 0.0)
        print(f"Drift metrics collected: {drift_metrics}")
        print(f"Drift Analysis Done! Share: {current_drift_share:.2%}")

        # Save HTML locally
        # report_html_path = "artifacts/monitoring/drift_report.html"
        # os.makedirs(os.path.dirname(report_html_path), exist_ok=True)
        # report.save_html(report_html_path)
        
        # ---- Artifacts dir ----
        monitoring_dir = "artifacts/monitoring"
        os.makedirs(monitoring_dir, exist_ok=True)
        
        # ---- MLflow run ----
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        # ใช้ nested=True เพื่อให้ไปอยู่ภายใต้วงจร Pipeline หลัก
        with mlflow.start_run(run_name=f"Monitoring_Drift_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}", nested=True):

            # ✅ 1. Log drift metrics เป็นตัวเลข — โชว์ใน Model metrics tab
            mlflow.log_metrics(drift_metrics)
 
            # ✅ 2. Per-column drift bar chart — render ใน Artifacts tab
            if per_column_drift:
                drift_plot_path = self._save_column_drift_plot(per_column_drift, monitoring_dir)
                mlflow.log_artifact(drift_plot_path, artifact_path="plots")
 
            # ✅ 3. retrain_triggered.json — audit trail
            retrain_result = {
                "timestamp":          pd.Timestamp.now().isoformat(),
                "drift_share":        current_drift_share,
                "drifted_columns":    drift_metrics.get("drifted_columns", 0),
                "total_columns":      drift_metrics.get("total_columns", 0),
                "dataset_drift_detected": bool(drift_metrics.get("dataset_drift_detected", 0)),
                "target_drift_detected":  bool(drift_metrics.get("target_drift_detected", 0)),
                "retrain_triggered":  current_drift_share > 0.40,
            }
            
            retrain_path = os.path.join(monitoring_dir, "retrain_triggered.json")
            with open(retrain_path, "w") as f:
                json.dump(retrain_result, f, indent=2)
            mlflow.log_artifact(retrain_path, artifact_path="monitoring")
 
            # ✅ 4. per_column_drift.json — รายละเอียดต่อ column
            cols_drift_path = os.path.join(monitoring_dir, "per_column_drift.json")
            with open(cols_drift_path, "w") as f:
                json.dump(per_column_drift, f, indent=2)
            mlflow.log_artifact(cols_drift_path, artifact_path="monitoring")
 
            # ✅ 5. drift_report.html — ดาวน์โหลดดูเองได้ (ไม่ render ใน UI)
            report_html_path = os.path.join(monitoring_dir, "drift_report.html")
            report.save_html(report_html_path)
            mlflow.log_artifact(report_html_path, artifact_path="drift_reports")
 
            if current_drift_share > 0.40:
                print("Warning: High Drift detected! Recommendation: Retrain Model.")
                return True
            return False
 
    def _save_column_drift_plot(self, per_column_drift: dict, output_dir: str) -> str:
        """สร้าง bar chart แสดง drift score ต่อ column พร้อม highlight column ที่ drift"""
        cols   = list(per_column_drift.keys())
        scores = [per_column_drift[c]["drift_score"] for c in cols]
        drifted = [per_column_drift[c]["drift_detected"] for c in cols]
 
        colors = ['tomato' if d else 'steelblue' for d in drifted]
 
        fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.9), 5))
        bars = ax.bar(cols, scores, color=colors, edgecolor='white', linewidth=0.5)
 
        # threshold line (p-value 0.05 หรือ ค่า drift score ที่ evidently ใช้)
        ax.axhline(y=0.05, color='black', linestyle='--', lw=1, label='Threshold 0.05')
 
        ax.set_xlabel("Feature")
        ax.set_ylabel("Drift score (p-value / distance)")
        ax.set_title("Per-column drift score")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
 
        # legend สี
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='tomato',    label='Drift detected'),
            Patch(facecolor='steelblue', label='No drift'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
 
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "column_drift_scores.png")
        fig.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
 
        return plot_path

        