import pandas as pd
import onnxruntime as ort
import mlflow
import numpy as np
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.config.mlflow_config import (
    MLFLOW_CACHE_DIR,
    MODEL_ALIAS,
    MODEL_ARTIFACT_PATH,
    MODEL_NAME,
    MLFLOW_TRACKING_URI,
)
from typing import Tuple

class PredictModel:
    def __init__(self):     
        self._model_name = MODEL_NAME
        self._model_alias = MODEL_ALIAS
        self._model_version = None
        self._tracking_uri = MLFLOW_TRACKING_URI
        
        mlflow.set_tracking_uri(self._tracking_uri)
        self._client = MlflowClient()
        
        self.cache_dir = Path(MLFLOW_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._load_model()
    
    def _load_model(self) -> ort.InferenceSession:
        """Загрузка модели из Mlflow"""
        try:
            model_version_info = self._client.get_model_version_by_alias(
                name=self._model_name,
                alias=self._model_alias,
            )
            self._model_version = model_version_info.version

            local_path = mlflow.artifacts.download_artifacts(
                run_id=model_version_info.run_id,
                artifact_path=MODEL_ARTIFACT_PATH,
                dst_path=str(self.cache_dir),
            )

            local_dir = Path(local_path)
            onnx_files = list(local_dir.rglob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"[PredictModel] .onnx не найден в {local_dir}")

            return ort.InferenceSession(str(onnx_files[0]))
        except Exception as e:
            raise RuntimeError(f"[PredictModel] Ошибка загрузки ONNX модели: {e}") from e
        
    def _fit_predict(self, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Внутренний метод для прогноза через ONNX Runtime"""
        input_name = self.model.get_inputs()[0].name
        X_arr = X_test.to_numpy(dtype=np.float32)

        outputs = self.model.run(None, {input_name: X_arr})
        labels = outputs[0]
        proba = outputs[1]

        if isinstance(proba, list) and len(proba) > 0 and isinstance(proba[0], dict):
            preds_proba = np.array([d.get(1, 0.0) for d in proba])
        else:
            proba_arr = np.asarray(proba)
            preds_proba = proba_arr[:, 1] if proba_arr.ndim == 2 and proba_arr.shape[1] > 1 else proba_arr.ravel()

        predictions = np.asarray(labels).ravel()
        return predictions, preds_proba
    
    def fit_predict(self, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Публичный метод для прогноза"""
        return self._fit_predict(X_test)

        
        