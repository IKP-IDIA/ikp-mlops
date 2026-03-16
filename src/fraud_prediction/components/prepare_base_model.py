import os 
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf 
from pathlib import Path
from fraud_prediction.entity.config_entity import PrepareBaseModelConfig
from fraud_prediction import logger

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        input_dim = self.config.params_num_features
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
        # เรียกใช้ save_model ผ่าน self
        self.save_model(path=self.config.base_model_path, model=self.model)

    def update_base_model(self):
        full_model = tf.keras.models.Sequential([
            self.model,
            tf.keras.layers.Dense(self.config.params_classes, activation='sigmoid')
        ])
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        full_model.summary()
        self.save_model(path=self.config.update_base_model_path, model=full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        logger.info(f"Model saved at: {path}")