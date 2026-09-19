import io
import pandas as pd
from src.storage.data_manager import DataManager
from src.monitoring.metrics import STAGE_LATENCY, PIPELINE_ERRORS

def get_data_manager(app) -> DataManager:
    return app.state.data_manager

def load_data(dm: DataManager, key: str) -> pd.DataFrame:
    """Получение даннных из хранилища"""
    try:
        with STAGE_LATENCY.labels(stage="read_data_from_s3").time():
            data = dm.read_data(key)
            data = pd.read_csv(io.BytesIO(data))
            
            if data.empty:
                raise ValueError("[load_data] data can`t be empty")
            return data
    
    except Exception as e:
        PIPELINE_ERRORS.labels(stage="unknown", error_type=type(e).__name__).inc()
        raise

def write_data_to_s3(dm: DataManager, key:str, data: pd.DataFrame) -> None:
    """Запись данных в хранилище"""
    try:
        with STAGE_LATENCY.labels(stage="write_data_to_s3").time():
            if key is None:
                raise ValueError("[write_data_to_s3] key can`t be None")
            if data.empty:
                raise ValueError("[write_data_to_s3] data can`t be empty")
            
            csv_bytes = data.to_csv(index=False).encode('utf-8')
        
            dm.write_data(csv_bytes, key)
            print("[write_data_to_s3] Data has recorded in S3 storage")
    
    except Exception as e:
        PIPELINE_ERRORS.labels(stage="unknown", error_type=type(e).__name__).inc()
        raise