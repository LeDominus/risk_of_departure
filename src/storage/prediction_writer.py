import uuid
import pandas as pd
from datetime import datetime, timezone
from src.storage.data_manager import DataManager
from src.config.config import PREDICTIONS_PREFIX
from src.utils.buffer import buffer_data

def append_pred_log_to_s3(
    dm: DataManager,
    prepared_df: pd.DataFrame,
    result_df: pd.DataFrame | None = None,
    model_version: str = "unknown",
) -> None:
    """Запись логов прогнозов в S3. Не пробрасывает исключения."""
    if prepared_df is None or prepared_df.empty:
        print("[predictions_writer] prepared_df пуст, пропускаем", flush=True)
        return

    now = datetime.now(timezone.utc)
    date_part = now.strftime("%Y-%m-%d")
    ts = now.strftime("%H%M%S_%f")

    df = prepared_df.copy()

    # Единое имя ID-колонки
    if "customerID" in df.columns and "customer_id" not in df.columns:
        df = df.rename(columns={"customerID": "customer_id"})

    df["prediction_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["prediction_date"] = now
    df["model_version"] = model_version

    try:
        # ---------- 1. Приводим предсказания к единому виду ----------
        preds = _normalize_predictions(result_df)
        print(
            f"[predictions_writer] prepared={df.shape}, preds={preds.shape}, "
            f"preds cols={preds.columns.tolist()}",
            flush=True,
        )

        # ---------- 2. Приклеиваем предсказания ----------
        if not preds.empty:
            if "customer_id" in df.columns and "customer_id" in preds.columns:
                # Идеальный случай: мёрж по ID
                df = df.merge(
                    preds.drop(columns=["customer_id"]),
                    left_on="customer_id",
                    right_index=True,
                    how="left",
                ) if False else df.merge(preds, on="customer_id", how="left")
            elif len(preds) == len(df):
                # Fallback: порядок сохранён
                df = pd.concat(
                    [df.reset_index(drop=True),
                     preds.drop(columns=["customer_id"], errors="ignore").reset_index(drop=True)],
                    axis=1,
                )
                print("[predictions_writer] merge по порядку строк", flush=True)
            else:
                print(
                    f"[predictions_writer] merge невозможен: "
                    f"df={len(df)}, preds={len(preds)}, "
                    f"customer_id в df: {'customer_id' in df.columns}",
                    flush=True,
                )
        else:
            print("[predictions_writer] preds пуст — пишем только фичи", flush=True)

        # ---------- 3. Типы (безопасно) ----------
        df = _format_types(
            data=df,
            types_dict={"churn": "Int8", "probability": "float32"},
        )

        # ---------- 4. Запись в S3 ----------
        buf = buffer_data(df=df)
        key = f"{PREDICTIONS_PREFIX}/prediction_date={date_part}/part-{ts}.parquet"
        dm.write_data(buf.getvalue(), key=key)

        print(f"[predictions_writer] wrote {len(df)} rows -> {key}", flush=True)

    except Exception as e:
        import traceback
        print(f"[predictions_writer] ошибка записи в S3: {e}", flush=True)
        traceback.print_exc()


def _normalize_predictions(result_df) -> pd.DataFrame:
    """Приводит result к DataFrame с колонками [customer_id, churn, probability]."""
    if result_df is None:
        return pd.DataFrame()

    if not isinstance(result_df, pd.DataFrame):
        if isinstance(result_df, dict) and result_df.get("data"):
            result_df = pd.DataFrame(result_df["data"])
        else:
            return pd.DataFrame()

    preds = result_df.copy()
    rename_map = {
        "customerID": "customer_id",
        "Churn": "churn",
        "Probability_of_churn": "probability",
    }
    preds = preds.rename(columns={k: v for k, v in rename_map.items() if k in preds.columns})

    keep = [c for c in ["customer_id", "churn", "probability"] if c in preds.columns]
    return preds[keep]
        
def _format_types(data: pd.DataFrame, types_dict: dict) -> pd.DataFrame:
    df = data.copy()
    
    for col, type in types_dict.items():
        df[col] = df[col].astype(type)
    
    return df
    