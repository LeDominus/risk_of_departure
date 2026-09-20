import json
import time
import pandas as pd
from prometheus_client import start_http_server, Gauge, Histogram
from evidently import Report
from evidently.presets import DataDriftPreset
from src.config.monitoring_config import LOG_PATH, REFERENCE_PATH, WINDOW_ROWS, INTERVAL_SECONDS, MIN_SAMPLE_SIZE
from src.storage.duckdb_manager import DuckDBManager
from src.schemas.feature_schema import FEATURE_COLS

DRIFT_SHARE = Gauge(
    "evidently_drift_share",
    "Share of drifted features",
)
PSI_SCORE = Gauge(
    "evidently_psi_score",
    "Drift score for a feature",
    ["feature"],
)
CURRENT_ROWS = Gauge(
    "evidently_current_rows",
    "Rows in current window"
)
S3_QUERY_LATENCY = Histogram(
    "evidently_s3_query_seconds",
    "DuckDB S3 query latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)

def load_reference() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE_PATH)
    df = df[[c for c in FEATURE_COLS if c in df.columns]]
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

def run_monitoring(reference_data: pd.DataFrame, db: DuckDBManager) -> None:
    with S3_QUERY_LATENCY.time():
        df = db.parse_logs(limit=WINDOW_ROWS)
    
    if df is None or df.empty:
        print("[evidently] нет предсказаний в S3", flush=True)
        CURRENT_ROWS.set(0)
        return
    CURRENT_ROWS.set(len(df))
    
    if len(df) < MIN_SAMPLE_SIZE:
        print(
            f"[evidently] пропускаем: строк {len(df)} < {MIN_SAMPLE_SIZE}",
            flush=True,
        )
        return
    
    common = [c for c in FEATURE_COLS if c in df.columns and c in reference_data.columns]
    if not common:
        print("[evidently] нет общих колонок между current и reference", flush=True)
        return
    
    current = df[common]
    ref = reference_data[common]

    print(f"[evidently] current rows: {len(current)}, features: {len(common)}", flush=True)
    
    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=current, reference_data=ref)
    result = my_eval.dict()

    drift_share = extract_drift_share(result)
    if drift_share is not None:
        DRIFT_SHARE.set(float(drift_share))
        print(f"[evidently] drift_share = {drift_share:.4f}", flush=True)
    else:
        print("[evidently] drift_share не найден", flush=True)

    psi_scores = extract_psi_scores(result)
    for feature, score in psi_scores.items():
        PSI_SCORE.labels(feature=feature).set(score)
    print(f"[evidently] psi computed for {len(psi_scores)} features", flush=True)

#TODO: потом в отдельный микросервис
if __name__ == "__main__":
    start_http_server(8000)
    print("[evidently] Prometheus exporter запущен на :8000", flush=True)

    reference_data = load_reference()

    with DuckDBManager() as db:
        while True:
            try:
                run_monitoring(reference_data, db)
            except Exception as e:
                import traceback
                print(f"[evidently] Ошибка: {e}", flush=True)
                traceback.print_exc()
            time.sleep(INTERVAL_SECONDS)