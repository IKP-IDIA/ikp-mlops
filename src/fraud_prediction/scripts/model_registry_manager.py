import mlflow

class ModelRegistryManager:
    def __init__(self, model_name="FraudDetection_Model"):
        # ตั้งค่า URI ก่อนสร้าง Client เพื่อให้คุยกับเครื่อง .150 ได้
        mlflow.set_tracking_uri("http://10.1.0.150:5000")
        self.client = mlflow.tracking.MlflowClient()
        self.model_name = model_name

    def promote_to_production(self, version: int):
        """เปลี่ยนสถานะโมเดลเป็น Production และเก็บเวอร์ชันเก่าเข้า Archive"""
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True 
            )
            print(f"✅ Success: Model {self.model_name} v{version} is now in Production.")
        except Exception as e:
            print(f"❌ Error: {e}")

    def get_latest_production_version(self):
        """ดึงข้อมูลเวอร์ชันที่อยู่ใน Production ปัจจุบัน"""
        versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
        if versions:
            return versions[0].version
        return None