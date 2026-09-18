import io

import pandas as pd
from src.storage.data_manager import DataManager

def get_data_manager(app) -> DataManager:
    return app.state.data_manager

def load_data(dm: DataManager, key: str) -> pd.DataFrame:
    """Получение даннных из хранилища"""
    data = dm.read_data(key)
    data = pd.read_csv(io.BytesIO(data))
    
    if data.empty:
        raise ValueError("[load_data] data can`t be empty")
    return data