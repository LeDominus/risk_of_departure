import os
from dotenv import load_dotenv
load_dotenv()

AWS_SECRET_KEY_ID = os.getenv("AWS_SECRET_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET")
S3_ENDPOINT = os.getenv("S3_ENDPOINT")
PREDICTIONS_PREFIX = os.getenv("S3_PREDICTIONS_PREFIX", "risk-of-departure/predictions")
S3_REGION = os.getenv("S3_REGION", "ru-central1")