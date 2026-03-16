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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from fraud_prediction.entity.config_entity import EvaluationConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

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
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.X_valid = np.asarray(X_scaled).astype('float32')
        self.y_valid = np.asarray(y).astype('float32')

    @mlflow.trace(name="Full_Evaluation_Process")
    def evaluation(self):
        # 1. Load model 
        self.model = self.load_model(self.config.path_of_model)
        self._prepare_validation_data()
        
        #Predic class (0 or 1)
        y_pred_prob = self.model.predict(self.X_valid)
        y_pred = (y_pred_prob > 0.5).astype(int)

        y_pred_prob = self.model.predict(self.X_valid)
        self.y_pred = (y_pred_prob > 0.5).astype(int)

        # Measure results by using Data Array
        self.recall = recall_score(self.y_valid, y_pred)
        self.precision = precision_score(self.y_valid, y_pred)
        self.f1 = f1_score(self.y_valid, y_pred)
        self.score = self.model.evaluate(self.X_valid, self.y_valid) # [loss, accuracy]
        
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

def log_into_mlflow(self, experiment_name=None):
        """ส่งผลลัพธ์ขึ้น Local MLflow (.150)"""
        
        # 1. ตั้งค่าการเชื่อมต่อ (ดึง URI จาก Config ที่เราแก้เป็น 10.1.0.150:5000)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        
        # กำหนดชื่อ Experiment (ใช้จาก Config หรือ Argument)
        exp_name = experiment_name if experiment_name else self.config.experiment_name
        mlflow.set_experiment(exp_name)

        # เคลียร์ Active Run เก่าถ้ามี
        try:
            if mlflow.active_run():
                mlflow.end_run()
        except Exception:
            pass

        now = datetime.now().strftime("%Y%m%d_%H%M")
        auto_run_name = f"Fraud_Pipeline_{now}"

        with mlflow.start_run(run_name=auto_run_name):
            # 2. บันทึก Parameters
            params = self.config.all_params
            mlflow.log_params({
                "epochs": int(params.EPOCHS),
                "batch_size": int(params.BATCH_SIZE),
                "learning_rate": float(params.LEARNING_RATE),
                "num_features": int(params.NUM_FEATURES)
            })
            
            # 3. บันทึก Metrics (เอาทั้ง Accuracy และ F1/Recall)
            report = classification_report(self.y_valid, self.y_pred, output_dict=True)
            mlflow.log_metrics({
                "loss": float(self.score[0]), 
                "accuracy": float(self.score[1]),
                "eval_f1_fraud": report['1.0']['f1-score'],
                "eval_recall_fraud": report['1.0']['recall'],
                "eval_precision_fraud": report['1.0']['precision']
            })
            
            # 4. บันทึก Plots (Confusion Matrix)
            cm = confusion_matrix(self.y_valid, self.y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Fraud Detection')
            plot_path = "confusion_matrix.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path) # ไฟล์นี้จะไปอยู่ใน MinIO (.250) โดยอัตโนมัติ

            # 5. บันทึก Model และ Signature
            signature = infer_signature(self.X_valid, self.model.predict(self.X_valid))
            
            # บันทึกโมเดลเข้า Model Registry ใน MLflow
            mlflow.keras.log_model(
                model=self.model, 
                artifact_path="model", 
                registered_model_name="FraudDetection_Model",
                signature=signature
            )
            print(f"🚀 Success: Results and Model logged to MLflow at {self.config.mlflow_uri}")
