import json
import mlflow
from mlflow.tracking import MlflowClient
from src.config.mlflow_config import MLFLOW_TRACKING_URI

RUN_ID = ""

def upload_params() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    with open("src/scripts/model_params.json") as f:
        params = json.load(f)

    flat = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            flat[k] = v
        else:
            flat[k] = json.dumps(v)

    client.log_batch(
        run_id=RUN_ID,
        params=[mlflow.entities.Param(k, str(v)) for k, v in flat.items()],
    )
    print(f"[upload] {len(flat)} параметров записаны в run {RUN_ID}")


if __name__ == "__main__":  
    upload_params()