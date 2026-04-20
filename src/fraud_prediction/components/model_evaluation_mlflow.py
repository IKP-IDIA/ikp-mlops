import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from fraud_prediction.utils.common import save_json 
import os
import glob
import json
from mlflow.models.signature import infer_signature
from fraud_prediction.entity.config_entity import EvaluationConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    auc, precision_recall_curve, roc_curve, roc_auc_score
)
from datetime import datetime
import joblib


class Evaluation:
    def __init__(self, config: EvaluationConfig): 
        self.config = config

    @mlflow.trace(name="Model_Loading") 
    @staticmethod
    def load_model(self, path: StopAsyncIteration) -> tf.keras.Model:
        return tf.keras.models.load_model(str(path))
    
    @mlflow.trace(name="Data_Preparation")
    def _prepare_validation_data(self):
        """โหลดและเตรียมข้อมูลสำหรับวัดผล (ลอจิกเดียวกับตอนเทรน)"""
        data_dir = self.config.training_data
        print(f"DEBUG: กำลังค้นหาไฟล์ใน Path -> {os.path.abspath(data_dir)}")

        if str(data_dir).endswith('csv') and os.path.isfile(data_dir):
            target_file = data_dir
        
        else:
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            if not csv_files:
                csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
            if not csv_files:
                raise FileNotFoundError(f"Not found file .csv in: {os.path.abspath(data_dir)}")
            target_file = csv_files[0]
        
        print(f" Found data file: {target_file}")
        df = pd.read_csv(target_file)
        
        # 1. Feature Engineering (ต้องเหมือนตอนเทรน)
        df['diff_new_old_balance'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['diff_new_old_destiny'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud','step_weeks', 'step_days'] 
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        # Important to put dtype = int bc about error while doing One-Hot
        df = pd.get_dummies(df, columns=['type'], dtype=int)
        df = df.dropna()

        # 2. แยก X, y
        X = df.drop(columns=['isFraud'])
        y = df['isFraud']

        # 3. Scaling (ในระบบจริงควรโหลด scaler ที่ Save ไว้มาใช้ แต่ตอนนี้ทำใหม่เพื่อ Test)
        #scaler = MinMaxScaler()
        scaler_path = os.path.join(os.path.dirname(self.config.path_of_model), "scaler.pkl")
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        
        self.X_valid = np.asarray(X_scaled).astype('float32')
        self.y_valid = np.asarray(y).astype('float32')
        
    @mlflow.trace(name="Full_Evaluation_Process")
    def evaluation(self):
        # 1. Load model 
        self.model = self.load_model(self.config.path_of_model)
        self._prepare_validation_data()
        
        #Predic class (0 or 1)

        y_pred_prob = self.model.predict(self.X_valid)
        self.y_pred = (y_pred_prob > 0.50).astype(int)

        # Measure results by using Data Array
        self.recall = recall_score(self.y_valid, self.y_pred)
        self.precision = precision_score(self.y_valid, self.y_pred)
        self.f1 = f1_score(self.y_valid, self.y_pred)
        self.score = self.model.evaluate(self.X_valid, self.y_valid) # [loss, accuracy]
        
        self.save_score()
    
    
    def run_tuning(self, learning_rates=[0.01, 0.001, 0.0001], batch_sizes=[32, 64]):
        """ฟังก์ชันสำหรับ Tuning Hyperparameters และบันทึกผลแยกตาม Run"""
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
        # สร้าง Parent Run เพื่อคุมกลุ่มการจูนครั้งนี้
        with mlflow.start_run(run_name=f"Tuning_Session_{datetime.now().strftime('%m%d_%H%M')}"):
            for lr in learning_rates:
                for bs in batch_sizes:
                    # สร้าง Child Run (Nested) สำหรับแต่ละคู่ Parameter
                    with mlflow.start_run(run_name=f"Trial_LR_{lr}_BS_{bs}", nested=True):
                        # 1. ปรับค่า Optimizer (ตัวอย่างการ Tuning เฉพาะส่วน Evaluation)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                        
                        # 2. ประเมินผล
                        # หมายเหตุ: ในขั้นตอนนี้เรา Evaluate เท่านั้น ถ้าจะ Train ใหม่ต้องเรียก fit()
                        eval_score = self.model.evaluate(self.X_valid, self.y_valid, batch_size=bs, verbose=0)
                        
                        # 3. บันทึกผลลงใน MLflow Child Run
                        mlflow.log_params({"learning_rate": lr, "batch_size": bs})
                        mlflow.log_metrics({
                            "eval_loss": float(eval_score[0]),
                            "eval_accuracy": float(eval_score[1])
                        })
                        
                        print(f"✅ Trial LR={lr}, BS={bs} | Accuracy: {eval_score[1]:.4f}")

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self, experiment_name=None):
        """ส่งผลลัพธ์ขึ้น Local MLflow (.150)"""
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        exp_name = experiment_name if experiment_name else self.config.experiment_name
        mlflow.set_experiment(exp_name)

        now = datetime.now().strftime("%Y%m%d_%H%M")
        auto_run_name = f"Model_Evaluation_{now}"

        with mlflow.start_run(run_name=auto_run_name, nested=True) as run:
            #---- Paramter ----
            params = self.config.all_params
            mlflow.log_params({
                "epochs": int(params.EPOCHS),
                "batch_size": int(params.BATCH_SIZE),
                "learning_rate": float(params.LEARNING_RATE),
                "num_features": int(params.NUM_FEATURES)
            })
            
            # ---- Predict ----
            y_pred_prob = self.model.predict(self.X_valid)
            
            # ---- Plots (render ได้ใน MLflow UI) ----
            artifacts_dir = "artifacts/model_evaluation/plots"
            self._save_and_log_plots(y_pred_prob, artifacts_dir=artifacts_dir)
            
            # ---- Metrics ----
            report = classification_report(self.y_valid, self.y_pred, output_dict=True)
            self.fraud_stats = report.get('1', report.get('1.0', {}))
            
            roc_auc = roc_auc_score(self.y_valid, y_pred_prob)
 
            mlflow.log_metrics({
                "loss":               float(self.score[0]),
                "accuracy":           float(self.score[1]),
                "eval_f1_fraud":      float(self.fraud_stats.get('f1-score', 0.0)),
                "eval_recall_fraud":  float(self.fraud_stats.get('recall', 0.0)),
                "eval_precision_fraud": float(self.fraud_stats.get('precision', 0.0)),
                "roc_auc":            float(roc_auc),
            })
            
            # ✅ classification_report.json — artifact สำหรับดูรายละเอียดทุก class
            os.makedirs("artifacts/model_evaluation", exist_ok=True)
            report_path = "artifacts/model_evaluation/classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(local_path=report_path, artifact_path="evaluation")
            
            # scores.json
            scores ={"loss":float(self.score[0]), "accuracy":float(self.score[1])}
            scores_path = "artifacts/model_evaluation/scores.json"
            with open(scores_path, "w") as f:
                json.dump(scores, f, indent=2)
            mlflow.log_artifact(scores_path, artifact_path="evaluation")
            
            #----Register model----            
            signature = infer_signature(self.X_valid, y_pred_prob)
            mlflow.keras.log_model(
                model=self.model, 
                artifact_path="model", 
                registered_model_name=self.config.registered_model_name,
                signature=signature
            )
            
            #----Quanlity Gate----
            current_recall = float(self.fraud_stats.get('recall', 0.0))
            if current_recall >= 0.60:
                print(f"✅ [PASSED] Recall: {current_recall:.2f} | Promoting to Production")
                self._promote_model_to_production()
            else:
                print(f"❌ [FAILED] Recall: {current_recall:.2f} | Deployment Stopped")

    def _save_and_log_plots(self, y_prob, artifacts_dir="artifacts/plots" , run_id=None):
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # ✅ Plot 1: Confusion Matrix
        cm = confusion_matrix(self.y_valid, self.y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        fig.savefig(cm_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")
 
        # ✅ Plot 2: Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(self.y_valid, y_prob)
        pr_auc = auc(recall_vals, precision_vals)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall_vals, precision_vals, color='steelblue', lw=2, label=f'PR AUC = {pr_auc:.3f}')
        ax.fill_between(recall_vals, precision_vals, alpha=0.1, color='steelblue')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        pr_path = os.path.join(artifacts_dir, "precision_recall_curve.png")
        fig.savefig(pr_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        mlflow.log_artifact(pr_path, artifact_path="plots")
        mlflow.log_metric("pr_auc", float(pr_auc))
 
        # ✅ Plot 3: ROC Curve (เพิ่มใหม่)
        fpr, tpr, _ = roc_curve(self.y_valid, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        roc_path = os.path.join(artifacts_dir, "roc_curve.png")
        fig.savefig(roc_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        mlflow.log_artifact(roc_path, artifact_path="plots")
        mlflow.log_metric("roc_auc", float(roc_auc))
 
        # ✅ Plot 4: Score distribution (prob histogram แยก fraud vs normal)
        fig, ax = plt.subplots(figsize=(7, 5))
        y_prob_flat = y_prob.flatten()
        ax.hist(y_prob_flat[self.y_valid == 0], bins=50, alpha=0.6, color='steelblue', label='Normal')
        ax.hist(y_prob_flat[self.y_valid == 1], bins=50, alpha=0.6, color='tomato',    label='Fraud')
        ax.axvline(x=0.5, color='black', linestyle='--', lw=1, label='Threshold 0.5')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        dist_path = os.path.join(artifacts_dir, "score_distribution.png")
        fig.savefig(dist_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        mlflow.log_artifact(dist_path, artifact_path="plots")
            
    def _promote_model_to_production(self):
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        model_name = self.config.registered_model_name
        
        versions = client.get_latest_versions(model_name)
        if not versions:
            print(f"No versions found for {model_name}. Skipping promotion.")

        latest_v = versions[0].version
        #latest_versions = client.get_latest_versions(model_name, stages=[None])[0].version
        client.transition_model_version_stage(
            name = model_name,
            version=latest_v,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model {model_name} version {latest_v} is now in PRODUCTION")
        
    