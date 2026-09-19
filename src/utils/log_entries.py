import json
from datetime import datetime
import pandas as pd

def log_prediction_entry(prepared_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    """Логгирование фактических предсказаний для мониторинга"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": prepared_df.to_dict(orient="records"),
        "prediction": result_df["Churn"]
    }
    
    #TODO: пока локально, затем перенести логику сохранения в отдельное место
    with open("data/logs/prediction_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")