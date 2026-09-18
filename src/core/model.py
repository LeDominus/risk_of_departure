import pandas as pd
from pathlib import Path
import onnxruntime as ort
import mlflow
import numpy as np
from src.config.config import MODEL_PARAMS
from typing import Tuple

MODEL_NAME = "logistic_regression.onnx"
EXPERIMENT_NAME = "Default"
CACHE_DIR = "tmp/mlflow_cache/"

class PredictModel:
    def __init__(self):     
        self.experiment_name = EXPERIMENT_NAME
        self.model_name = MODEL_NAME
        self._model_params = MODEL_PARAMS
        self.run_id = None
        
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
        try:
            self._get_run_id()
        except Exception as e:
            raise ValueError(f"Error getting run_id: {e}")
        
        self.model = self._load_model()
        self._model_params = MODEL_PARAMS
    
    def _get_run_id(self) -> None:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_role = 'logreg_prod'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if len(runs) == 0:
            raise ValueError("Нет run'а с тегом model_role=logreg_prod")
        self.run_id = runs.iloc[0]["run_id"]
    
    def _load_model(self) -> ort.InferenceSession:
        try:
            artifact_path = f"model/{self.model_name}"
            print(f"DEBUG artifact_path={artifact_path!r}")

            local_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path=artifact_path,
                dst_path=str(self.cache_dir),
            )
            print(f"DEBUG local_path={local_path}")
            return ort.InferenceSession(local_path)
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

        
        