import os
import zipfile
import shutil
from fraud_prediction import logger
from fraud_prediction.utils.common import get_size
from fraud_prediction.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

     
    def download_file(self) -> str:
        try: 
            # ต้นทางคือไฟล์ที่เราทำ DVC ไว้ (เช่น data/fraud_0.1origbase.csv)
            source_file = self.config.local_data_file 
            # ปลายทางใน artifacts
            dest_dir = self.config.unzip_dir
            
            os.makedirs(dest_dir, exist_ok=True)
            
            logger.info(f"Using local data from DVC: {source_file}")
            
            # ตรวจสอบว่ามีไฟล์ต้นทางอยู่จริงไหม
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"หาไฟล์ {source_file} ไม่เจอ! กรุณารัน dvc pull ก่อน")

            # คัดลอกไฟล์ไปไว้ใน artifacts เพื่อให้ stage อื่นใช้งานต่อ
            dest_path = os.path.join(dest_dir, os.path.basename(source_file))
            shutil.copy(source_file, dest_path)
            
            logger.info(f"Successfully copied data to {dest_path}")
            return dest_path

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        ถ้าไฟล์ที่คัดลอกมาเป็น .csv อยู่แล้ว ก็ไม่ต้องทำอะไร 
        แต่ถ้าเป็น .zip ให้แตกไฟล์ตามปกติ
        """
        source_file = self.config.local_data_file
        unzip_path = self.config.unzip_dir
        
        if source_file.endswith('.zip'):
            logger.info(f"Extracting zip file: {source_file}")
            import zipfile
            with zipfile.ZipFile(source_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        else:
            logger.info("File is not a zip, skipping extraction.")

