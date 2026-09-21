import mlflow
import onnx
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.config.mlflow_config import MLFLOW_TRACKING_URI

MODEL_PARAMS = {
    "n_jobs": -1,
    "verbose": 100,
    "max_iter": 100,
    "class_weight": "balanced",
}
MLFLOW_MODEL_NAME = "logreg_regression"
ONNX_MODEL_PATH = Path("data/result/logistic_regression.onnx")

def registry_model() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print(f"[registry] loading ONNX from {ONNX_MODEL_PATH}", flush=True)
    onnx_model = onnx.load(str(ONNX_MODEL_PATH))
    print(f"[registry] ONNX loaded: ir_version={onnx_model.ir_version}", flush=True)
    
    with mlflow.start_run(run_name="logreg_training") as run:
        mlflow.log_params(MODEL_PARAMS)

        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        print(f"[registry] Run ID: {run.info.run_id}", flush=True)
        print(f"[registry] Model '{MLFLOW_MODEL_NAME}' registered", flush=True)
    
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))

    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias="champion",
        version=latest.version,
    )
    print(f"[registry] @champion → v{latest.version}", flush=True)
    
if __name__ == "__main__":
    registry_model()