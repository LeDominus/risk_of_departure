import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from prometheus_client import start_http_server, Gauge
from evidently import Report
from evidently.presets import DataDriftPreset

LOG_PATH = Path("data/logs/prediction_logs.jsonl")
REFERENCE_PATH = Path("data/raw/churn_data_new.csv")
INTERVAL_SECONDS = 30
WINDOW_ROWS = 1000
PREDICTION_LOG_PATH = Path("data/logs/prediction_logs.jsonl")

DRIFT_SHARE = Gauge(
    "evidently_drift_share",
    "Share of drifted features",
)
PSI_SCORE = Gauge(
    "evidently_psi_score",
    "Drift score for a feature",
    ["feature"],
)

def append_prediction_log(prepared_df: pd.DataFrame, result: dict | None = None) -> None:
    try:
        if prepared_df is None or prepared_df.empty:
            return

        path = PREDICTION_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat()

        predictions_by_id = {}
        if isinstance(result, dict):
            for item in result.get("data", []):
                cid = item.get("customerID")
                if cid is not None:
                    predictions_by_id[cid] = {
                        "Churn": item.get("Churn"),
                        "Probability_of_churn": item.get("Probability_of_churn"),
                    }

        with path.open("a", encoding="utf-8") as f:
            for record in prepared_df.to_dict(orient="records"):
                entry = {
                    "timestamp": timestamp,
                    "features": record,
                }

                # Если есть customerID и для него есть предсказание — добавим
                cid = record.get("customerID")
                if cid is not None and cid in predictions_by_id:
                    entry["prediction"] = predictions_by_id[cid]

                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    except Exception as e:
        # Логирование не должно ломать основной пайплайн
        print(f"[append_prediction_log] Ошибка логирования: {e}", flush=True)

def load_reference() -> pd.DataFrame:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference не найден: {REFERENCE_PATH}")
    df = pd.read_csv(REFERENCE_PATH)
    print(f"[evidently] reference loaded: {df.shape}", flush=True)
    return df

def load_current() -> pd.DataFrame:
    if not LOG_PATH.exists():
        print(f"[evidently] {LOG_PATH} ещё не создан", flush=True)
        return pd.DataFrame()

    rows = []
    with LOG_PATH.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        print("[evidently] Файл логов пуст", flush=True)
        return pd.DataFrame()

    rows = rows[-WINDOW_ROWS:]
    features = [r["features"] for r in rows if "features" in r]
    if not features:
        print("[evidently] В логах нет ключа 'features'", flush=True)
        return pd.DataFrame()

    print(f"[evidently] current rows: {len(features)}", flush=True)
    return pd.DataFrame(features)

def extract_drift_share(result: dict):
    for metric in result.get("metrics", []):
        name = metric.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            if isinstance(value, dict):
                return value.get("share")
    return None

def extract_psi_scores(result: dict):
    scores = {}
    for metric in result.get("metrics", []):
        name = metric.get("metric_name", "")
        if not name.startswith("ValueDrift"):
            continue
        config = metric.get("config", {})
        column = config.get("column")
        value = metric.get("value")
        if column is not None and value is not None:
            try:
                scores[column] = float(value)
            except (TypeError, ValueError):
                continue
    return scores

def run_monitoring(reference_data: pd.DataFrame) -> None:
    current_data = load_current()
    if current_data.empty:
        return

    common_cols = [c for c in reference_data.columns if c in current_data.columns]
    if not common_cols:
        print("[evidently] Нет общих колонок между reference и current", flush=True)
        return

    ref = reference_data[common_cols]
    cur = current_data[common_cols]

    # --- Новый API: Report([preset]), run() возвращает my_eval ---
    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=cur, reference_data=ref)

    result = my_eval.dict()

    # Диагностика (можно закомментировать после отладки)
    print(f"[evidently] result keys: {list(result.keys())}", flush=True)

    # --- drift_share ---
    drift_share = extract_drift_share(result)
    if drift_share is not None:
        DRIFT_SHARE.set(float(drift_share))
        print(f"[evidently] drift_share = {drift_share:.4f}", flush=True)
    else:
        print("[evidently] Не удалось извлечь drift_share", flush=True)

    # --- PSI по колонкам ---
    psi_scores = extract_psi_scores(result)
    if psi_scores:
        for feature, score in psi_scores.items():
            PSI_SCORE.labels(feature=feature).set(score)
            print(f"[evidently] psi {feature} = {score:.4f}", flush=True)
    else:
        print("[evidently] Не удалось извлечь PSI-скоры", flush=True)

#TODO: потом в отдельный микросервис
if __name__ == "__main__":
    start_http_server(8000)
    print("[evidently] Prometheus exporter запущен на :8000", flush=True)

    reference_data = load_reference()

    while True:
        try:
            run_monitoring(reference_data)
        except Exception as e:
            import traceback
            print(f"[evidently] Ошибка: {e}", flush=True)
            traceback.print_exc()
        time.sleep(INTERVAL_SECONDS)