import pandas as pd
from src.core.model import PredictModel
from src.core.data_preparation import DataPreparation

def forecast_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """Pipeline для подготовки данных, инференса ONNX-модели и постпроцессинга"""
    if data.empty:
        raise ValueError("[forecast_pipeline] Входной DataFrame пуст")
        
    predict_model = PredictModel()
    data_preparation = DataPreparation(data)
    
    prepared_df = data_preparation.prepare_data()
    predictions, predictions_proba = predict_model.fit_predict(prepared_df)
    
    result = data_preparation.postprocess_predictions(
        predictions=predictions, 
        predictions_proba=predictions_proba
    )
    
    if result.empty:
        raise ValueError("[forecast_pipeline] Результат предсказания не может быть пустым")
    
    return result

    
    
    