from pathlib import Path

import mlflow

from src.config.mlflow_config import (
    MLFLOW_CACHE_DIR,
    MLFLOW_TRACKING_URI,
)

RUN_ID = ""
MLFLOW_MODEL_NAME = "logreg_regression"

def download_model() -> Path:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.get_run(RUN_ID)
    cache_dir = Path(MLFLOW_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=RUN_ID,
        artifact_path=f"model/{MLFLOW_MODEL_NAME}",
        dst_path=str(cache_dir),
    )
    return Path(local_path)


if __name__ == "__main__":
    output = download_model()
    print(output)