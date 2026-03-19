import os
import pandas as pd
import numpy as np
import mlflow 
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from fraud_prediction.entity.config_entity import TrainingConfig
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import recall_score, precision_score, f1_score
from fraud_prediction import logger
import joblib

class Training: 
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        """โหลด model ANN ที่เตรียมไว้จาก stage_02"""
        self.model = tf.keras.models.load_model(
            self.config.update_base_model_path
        )

        # Re-compile immediatly with new optimizer 
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model DownModel Downloaded and Re-Compiled already.")

    def prepare_data(self):
        """เตรียมข้อมูลตามลอจิก Notebook"""
        # 1. โหลดไฟล์ CSV
        data_dir = self.config.training_data
        csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
        if not csv_files:
            raise FileNotFoundError(f"ไม่พบไฟล์ CSV ใน {data_dir}")

        df = pd.read_csv(csv_files[0])

        fraud_df = df[df['isFraud'] == 1]
        normal_df = df[df['isFraud'] == 0]
        
        ratio = self.config.params_sampling_ratio
        n_normal=len(fraud_df)*ratio 
        # ตรวจสอบว่ามีข้อมูลปกติพอให้สุ่มไหม
        n_normal = min(n_normal, len(normal_df)) 
        
        normal_downsampled = normal_df.sample(n=n_normal, random_state=42)
        
        df = pd.concat([fraud_df, normal_downsampled])
        df = df.sample(frac=1, random_state=42)

        print(f"ข้อมูลที่ใช้เทรน (Ratio 1:{ratio}): ทั้งหมด {df.shape[0]} แถว")

        # 2. Feature Engineering
        df['diff_new_old_balance'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['diff_new_old_destiny'] = df['newbalanceDest'] - df['oldbalanceDest']

        # 3. Feature Selection & One-Hot Encoding
        cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud','step_weeks', 'step_days'] 
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

        # ป้องกัน NotImplementedError ด้วย dtype=int
        df = pd.get_dummies(df, columns=['type'], dtype=int)
        
        rows_before = df.shape[0]
        df = df.dropna()

        self.rows_after = df.shape[0]
        self.rows_lost = rows_before - self.rows_after
        
        logger.info(f"Data Cleaning: Remaining rows: {self.rows_after} (Lost: {self.rows_lost})")
        
        # 4. แยก Feature และ Target (isFraud)
        target_col = 'isFraud'
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 5. Seperate Data and Scaling
        X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        scaler_dir = os.path.dirname(self.config.trained_model_path)
        self.scaler_path = os.path.join(os.path.dirname(self.config.trained_model_path), "scaler.pkl")
        joblib.dump(scaler, self.scaler_path)
        X_valid_scaled = scaler.transform(X_valid_raw)

        # 6. บันทึกผลลัพธ์ลงใน Class Attribute (แปลงเป็น float32 ทันที)
        self.X_train = np.asarray(X_train_scaled).astype('float32')
        self.X_valid = np.asarray(X_valid_scaled).astype('float32')
        self.y_train = np.asarray(y_train_raw).astype('float32')
        self.y_valid = np.asarray(y_valid_raw).astype('float32')
        
        print(f" Prepared Data Done and จำนวน Features สุดท้าย: {self.X_train.shape[1]}")

    def train(self, experiment_name=None):
            """เริ่มเทรนโมเดลและ Log ผลลง MLflow (.150)"""
            
            # 1. ตั้งค่าการเชื่อมต่อให้ชัดเจน
            if hasattr(self.config, 'mlflow_uri'):
                mlflow.set_tracking_uri(self.config.mlflow_uri)
            
            # ใช้ชื่อ Experiment จาก Config หรือที่ส่งมา
            exp_name = experiment_name if experiment_name else self.config.experiment_name
            mlflow.set_experiment(exp_name)

            # แปลงข้อมูลเป็น Tensor
            X_train_tensor = tf.convert_to_tensor(self.X_train, dtype=tf.float32)
            y_train_tensor = tf.convert_to_tensor(self.y_train, dtype=tf.float32)
            X_valid_tensor = tf.convert_to_tensor(self.X_valid, dtype=tf.float32)
            y_valid_tensor = tf.convert_to_tensor(self.y_valid, dtype=tf.float32)
            
            # เปิด Autolog (จะเก็บ Loss/Acc ทุก Epoch ให้อัตโนมัติ)
            mlflow.keras.autolog(log_models=True)

            print(f"🚀 Starting training on MLflow: {mlflow.get_tracking_uri()}")
            
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )

            # 2. เริ่มต้นบันทึกผล
            with mlflow.start_run(run_name="Model_Training_Fit", nested=True):
                
                from sklearn.utils.class_weight import compute_class_weight
                mlflow.log_artifact(self.scaler_path)
                
                # หา class ที่มี (0 และ 1)
                classes = np.unique(self.y_train)
                # คำนวณน้ำหนักแบบ 'balanced' 
                # สูตร: n_samples / (n_classes * np.bincount(y))
                weights = compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=self.y_train
                )
                class_weights = dict(zip(classes, weights))
                
                print(f"⚖️ Calculated Class Weights: {class_weights}")
                # บันทึกน้ำหนักลง MLflow ด้วยเพื่อให้ตรวจสอบได้ย้อนหลัง
                mlflow.log_params({
                    "class_weight_0": class_weights[0],
                    "class_weight_1": class_weights[1]
                })
                
                # Log Parameters
                mlflow.log_params({
                    "epochs": self.config.params_epochs,
                    "batch_size": self.config.params_batch_size,
                    "sampling_ratio": self.config.params_sampling_ratio,
                    "input_features": self.X_train.shape[1]
                })
                mlflow.log_metric("training_rows", self.rows_after)
                mlflow.log_metric("rows_dropped_ratio", (self.rows_lost / (self.rows_after + self.rows_lost)))
                
                # สั่งเทรน
                self.history = self.model.fit(
                    X_train_tensor,
                    y_train_tensor,
                    epochs=self.config.params_epochs,
                    batch_size=self.config.params_batch_size,
                    validation_data=(X_valid_tensor, y_valid_tensor),
                    #class_weight=class_weights,
                    verbose=1
                )
                
                # คำนวณ Metrics หลังเทรนเสร็จ
                y_pred_prob = self.model.predict(X_valid_tensor)
                y_pred = (y_pred_prob > 0.5).astype(int)
                
                recall = recall_score(self.y_valid, y_pred)
                precision = precision_score(self.y_valid, y_pred, zero_division=0)
                f1 = f1_score(self.y_valid, y_pred, zero_division=0)
                
                # บันทึกลง MLflow
                mlflow.log_metrics({
                    "final_recall": recall, 
                    "final_precision": precision, 
                    "final_f1_score": f1
                })
                
                # บันทึกตัวโมเดลไว้ใน artifacts
                #mlflow.log_artifact(self.config.trained_model_path)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))