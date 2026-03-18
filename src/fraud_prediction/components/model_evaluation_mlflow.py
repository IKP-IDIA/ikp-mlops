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
        scaler = joblib.load("scaler.pkl")
        X_scaled = scaler.transform(X)
        
        self.X_valid = np.asarray(X_scaled).astype('float32')
        self.y_valid = np.asarray(y).astype('float32')
        
    import mlflow

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
        # 1. ตั้งค่าการเชื่อมต่อ (ดึง URI จาก Config ที่เราแก้เป็น 10.1.0.150:5000)
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_registry_uri(self.config.mlflow_uri)
        
            # กำหนดชื่อ Experiment (ใช้จาก Config หรือ Argument)
            exp_name = experiment_name if experiment_name else self.config.experiment_name
            mlflow.set_experiment(exp_name)

            # เคลียร์ Active Run เก่าถ้ามี
            # try:
            #     if mlflow.active_run():
            #         mlflow.end_run()
            # except Exception:
            #     pass
            

            now = datetime.now().strftime("%Y%m%d_%H%M")
            auto_run_name = f"Stage04_Eval_{now}"

            with mlflow.start_run(run_name=auto_run_name, nested=True):
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
                print(f"DEBUG Report Keys: {report.keys()}")
                
                self.fraud_stats = report.get('1', report.get('1.0', {}))
                
                mlflow.log_metrics({
                    "loss": float(self.score[0]), 
                    "accuracy": float(self.score[1]),
                    "eval_f1_fraud": self.fraud_stats.get('f1-score', 0.0),
                    "eval_recall_fraud": self.fraud_stats.get('recall', 0.0),
                    #"eval_precision_fraud": self.fraud_stats.get('precision', 0.0)
                })
                
                # 4. บันทึก Plots (Confusion Matrix)
                cm = confusion_matrix(self.y_valid, self.y_pred)
                plt.figure(figsize=(8,6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix - Fraud Detection')
                plot_path = "confusion_matrix.png" # ไฟล์ชั่วคราวในเครื่อง
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path) # บันทึกเข้า MLflow Artifacts
                plt.close()
                
                scores = {
                "loss": float(self.score[0]),
                "accuracy": float(self.score[1]),
                "recall": float(self.recall),
                "f1": float(self.f1)
                    }
                
                import json
                with open("eval_results.json", "w") as f:
                    json.dump(scores, f)
                mlflow.log_artifact("eval_results.json")
                
                # 5. บันทึก Model และ Signature
                signature = infer_signature(self.X_valid, self.model.predict(self.X_valid))
                
                # บันทึกโมเดลเข้า Model Registry ใน MLflow
                mlflow.keras.log_model(
                    model=self.model, 
                    artifact_path="model", 
                    registered_model_name=self.config.registered_model_name,
                    signature=signature
                )
                print(f"🚀 Success: Results and Model logged to MLflow at {self.config.mlflow_uri}")
                
                current_recall = self.fraud_stats.get('recall',0.0)
                current_loss = float(self.score[0])
                        
                THRESHOLD_RECALL = 0.70
                THRESHOLD_LOSS = 0
                        
                print(f"Checking Quality Gate: Recall={current_recall}, Loss={current_loss}")
                        
                if current_recall >=  THRESHOLD_RECALL and current_loss <= THRESHOLD_LOSS:
                    print("[PASSED] Model quality is good. Ready for Deployment")
                    self._promote_model_to_production()                
                else:
                    print("[FAIED] Model quanlity below threshold, Stopping deployment)")
                
    def _promote_model_to_production(self):
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        model_name = self.config.registered_model_name
        
        latest_versions = client.get_latest_versions(model_name, stages=[None])[0].version
        client.transition_model_version_stage(
            name = model_name,
            version=latest_versions,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Model {model_name} version {latest_versions} is now in PRODUCTION")
            
                
