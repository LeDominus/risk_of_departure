import os

TARGET_COLS = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges',
]

ORIGIN_COLS = TARGET_COLS.copy().append("customerID")

RUN_ID = "0e1218bb2329462fb7740069e5a4a161"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_MODEL_NAME = "logistic_regression.onnx"
MLFLOW_CACHE_DIR = "tmp/mlflow_cache/"

COLS_TO_CONV = ["MonthlyCharges", "TotalCharges"]
NUM_FEATURES = ["tenure", "MonthlyCharges", "SeniorCitizen", "TotalCharges"]

MODEL_PARAMS = {
    "n_jobs": -1,
    "verbose": 100,
    "max_iter": 100,
    "class_weight": "balanced",
    "random_state": 42
}
MODEL_PATH = "/data/result/logistic_regression.onnx"
RESULT_KEY = "forecast_result.csv"

from dotenv import load_dotenv
load_dotenv()

AWS_SECRET_KEY_ID = os.getenv("AWS_SECRET_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")