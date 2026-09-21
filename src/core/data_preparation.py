import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config.data_config import TARGET_COLS, COLS_TO_CONV

class DataPreparation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.origin_df = df
        self.le = LabelEncoder()
        self.target_cols = TARGET_COLS
    
    def _validate_input_data(self) -> None:
        """Валидация входнных данных"""
        missing = [col for col in self.df.columns if col not in self.target_cols]
        if missing:
            print(f"[DataPreparation] Input dataset has missing cols: {missing} remove it...")
            removed_df = self._leave_missing_cols(missing)
            print(f"[DataPreparation] Unneccessary cols were removed")
            self.df = removed_df
        else:
            print("[DataPreparation] Data is valid")

    def _leave_missing_cols(self, missing: list) -> pd.DataFrame:
        """Оставляем только целевые колонки для прогнозов"""
        df = self.df.copy()
        
        cols_for_remove = [col for col in df.columns if col in missing]
        df = df.drop(cols_for_remove, axis=1)
        
        if len(df.columns) != len(self.target_cols):
            raise ValueError(f"Несоотвествие числа колонок после удаления: {len(df.columns)}, {len(self.target_cols)}")
        
        if df.empty:
            raise ValueError(f"[DataPreparation] Input dataset after target_cols can`t be empty: {df.shape}")
        
        return df
    
    def _preprocess_data_types(self) -> pd.DataFrame:
        """Предобработка входных типов"""
        df = self.df.copy()
        
        for feature in COLS_TO_CONV:
            df[feature] = df[feature].replace(' ', np.nan)
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature] = df[feature].fillna(0)
        
        self.df = df
        print("Предобработка входных типов успешно завершена")
    
    def _encode_data_labels(self) -> pd.DataFrame:
        """Кодирование категориальных переменных"""
        df = self.df.copy()
        
        cat_features = df.select_dtypes(include=[str])
        for feature in cat_features:
            df[feature] = self.le.fit_transform(df[feature])
        
        self.df = df
        print("Кодирование категориальных переменных успешно завершено")
    
    def postprocess_predictions(self, predictions: pd.DataFrame, predictions_proba: pd.DataFrame) -> pd.DataFrame:
        """Публичный метод для постобработки прогнозов"""
        origin_df = self.origin_df.copy()
        
        result = pd.DataFrame({
            "customerID": origin_df['customerID'],
            "Churn": predictions,
            "Probability_of_churn": predictions_proba.round(4)
        })
        
        if result.empty:
            raise ValueError("[DataPreparation] result is empty")
        
        return result
    
    def prepare_data(self) -> pd.DataFrame:
        """Публичный метод для подготовки данных"""
                
        self._validate_input_data()
        self._preprocess_data_types()
        self._encode_data_labels()
        
        if self.df.empty:
            raise ValueError("[DataPreparation] df can`t be empty")
        
        return self.df
        