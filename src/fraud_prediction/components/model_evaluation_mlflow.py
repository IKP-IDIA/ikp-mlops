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
from mlflow.models.signature import infer_signature
from fraud_prediction.entity.config_entity import EvaluationConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, auc, confusion_matrix, ConfusionMatrixDisplay,precision_recall_curve
from datetime import datetime
import joblib

class Evaluation:
    def __init__(self, config: EvaluationConfig): 
        self.config = config

    @mlflow.trace(name="Model_Loading") 
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
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
        scaler_path = os.path.join(
            os.path.dirname(self.config.path_of_model), "scaler.pkl")
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
            # 1. บันทึก Parameters
            params = self.config.all_params
            mlflow.log_params({
                "epochs": int(params.EPOCHS),
                "batch_size": int(params.BATCH_SIZE),
                "learning_rate": float(params.LEARNING_RATE),
                "num_features": int(params.NUM_FEATURES)
            })
            
            # 2. ทำการ Predict เพื่อเอาค่า Probability (สำหรับ PR Curve)
            y_pred_prob = self.model.predict(self.X_valid)
            self._save_and_log_plots(y_pred_prob)
            
            # 3. บันทึก Metrics
            report = classification_report(self.y_valid, self.y_pred, output_dict=True)
            self.fraud_stats = report.get('1', report.get('1.0', {}))
            
            mlflow.log_metrics({
                "loss": float(self.score[0]), 
                "accuracy": float(self.score[1]),
                "eval_f1_fraud": float(self.fraud_stats.get('f1-score', 0.0)),
                "eval_recall_fraud": float(self.fraud_stats.get('recall', 0.0)),
            })
            
            # 5. บันทึก Model และ Signature
            signature = infer_signature(self.X_valid, y_pred_prob)
            mlflow.keras.log_model(
                model=self.model, 
                artifact_path="model", 
                registered_model_name=self.config.registered_model_name,
                signature=signature
            )
            
            # 6. Quality Gate Check
            current_recall = float(self.fraud_stats.get('recall', 0.0))
            if current_recall >= 0.60:
                print(f"✅ [PASSED] Recall: {current_recall:.2f} | Promoting to Production")
                self._promote_model_to_production()
            else:
                print(f"❌ [FAILED] Recall: {current_recall:.2f} | Deployment Stopped")

    def _save_and_log_plots(self, y_prob, artifacts_dir="artifacts/plots" , run_id=None):
        """Method สำหรับวาดกราฟและส่งขึ้น MLflow Artifacts"""
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Plot 1: Confusion Matrix
        cm = confusion_matrix(self.y_valid, self.y_pred)
        fig1, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title('Confusion Matrix')
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        fig1.savefig(cm_path, bbox_inches='tight')
        #mlflow.log_artifact(cm_path, artifact_path="plots") # เก็บไว้ในโฟลเดอร์ plots
        plt.close(fig1)

        # Plot 2: Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_valid, y_prob)
        pr_auc = auc(recall, precision)
        fig2, ax = plt.subplots(figsize=(8,6))
        ax.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        pr_path = os.path.join(artifacts_dir, "precision_recall_curve.png")
        fig2.savefig(pr_path, bbox_inches='tight')
        plt.close(fig2)
        
        #with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(pr_path, artifact_path="plots")
        mlflow.log_metric("pr_auc", float(pr_auc)) #bonus: log ค่่า AUC ด้วย
            
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
        
    