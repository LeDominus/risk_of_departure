from pathlib import Path

LOG_PATH = Path("data/logs/prediction_logs.jsonl")
REFERENCE_PATH = Path("data/raw/churn_data_new.csv")
INTERVAL_SECONDS = 30
WINDOW_ROWS = 1000
MIN_SAMPLE_SIZE = 300
PREDICTION_LOG_PATH = Path("data/logs/prediction_logs.jsonl")
