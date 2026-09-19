import pandas as pd
from src.core.model import PredictModel
from src.core.data_preparation import DataPreparation
from src.monitoring.metrics import STAGE_LATENCY, PIPELINE_ERRORS
from src.monitoring.metrics_registry import emit_batch_metrics
from src.monitoring.evidently_monitoring import append_prediction_log

def forecast_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """Pipeline для подготовки данных, инференса ONNX-модели и постпроцессинга"""
    if data.empty:
        raise ValueError("[forecast_pipeline] Входной DataFrame пуст")
        
    predict_model = PredictModel()
    data_preparation = DataPreparation(data)
    
    try:
        with STAGE_LATENCY.labels(stage="data_preparation").time():
            prepared_df = data_preparation.prepare_data()
        
        with STAGE_LATENCY.labels(stage="inference").time():
            predictions, predictions_proba = predict_model.fit_predict(prepared_df)
        
        with STAGE_LATENCY.labels(stage="postprocess_predictions").time():
            result = data_preparation.postprocess_predictions(
                predictions=predictions, 
                predictions_proba=predictions_proba
            )
        if result.empty:
            raise ValueError("[forecast_pipeline] Результат предсказания не может быть пустым")
        
    except Exception as e:
        PIPELINE_ERRORS.labels(stage="unknown", error_type=type(e).__name__).inc()
        raise
    
    finally:
        emit_batch_metrics(probs=predictions_proba)
        append_prediction_log(prepared_df=prepared_df)
    
    return result

    
    
    