import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from prometheus_client import start_http_server, Gauge
from evidently import Report
from evidently.presets import DataDriftPreset

DRIFT_SHARE = Gauge('evidently_drift_share', 'Share of drifted features')
PSI_SCORE = Gauge('evidently_psi_score', 'PSI score for a feature', ['feature'])
LOG_PATH = Path("data/logs/prediction_logs.jsonl")

reference_data = pd.read_csv("data/raw/churn_data_new.csv")

def append_prediction_log(prepared_df):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": prepared_df.to_dict(orient="records"),
    }
    with LOG_PATH.open("a") as f:
        for rec in entry["features"]:
            f.write(json.dumps({"timestamp": entry["timestamp"], "features": rec}) + "\n")

def run_monitoring():
    """Основной цикл мониторинга (новый API Evidently)."""
    try:
        with open("data/logs/prediction_logs.jsonl", "r") as f:
            logs = [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        print("[evidently] Файл логов ещё не создан", flush=True)
        return

    if not logs:
        print("[evidently] Файл логов пуст", flush=True)
        return

    rows = logs[-1000:]
    features = [r["features"] for r in rows if "features" in r]
    if not features:
        print("[evidently] В логах нет ключа 'features'", flush=True)
        return

    current_data = pd.DataFrame(features)

    # Пересекаем колонки
    common_cols = [c for c in reference_data.columns if c in current_data.columns]
    if not common_cols:
        print("[evidently] Нет общих колонок между reference и current", flush=True)
        return

    ref = reference_data[common_cols]
    cur = current_data[common_cols]

    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=cur, reference_data=ref)

    # --- ДИАГНОСТИКА: смотрим реальную структуру ---
    result = my_eval.dict()
    print("[evidently] Верхнеуровневые ключи result:", list(result.keys()), flush=True)
    if "metrics" in result and result["metrics"]:
        first_metric = result["metrics"][0]
        print("[evidently] Ключи первой метрики:", list(first_metric.keys()), flush=True)
        # Если внутри есть 'result', посмотрим и его ключи
        if "result" in first_metric:
            print("[evidently] Ключи result первой метрики:", list(first_metric["result"].keys()), flush=True)
    # --- КОНЕЦ ДИАГНОСТИКИ ---

    # --- Безопасное извлечение drift_share ---
    drift_share = None
    try:
        # Пробуем разные возможные пути
        metrics = result.get("metrics", [])
        if metrics:
            first = metrics[0]
            # Вариант 1: новый API с ключом 'result'
            if "result" in first:
                drift_share = first["result"].get("share_of_drifted_columns") or first["result"].get("drift_share")
            # Вариант 2: ключ 'value' (как в некоторых версиях Snapshot)
            elif "value" in first and isinstance(first["value"], dict):
                drift_share = first["value"].get("share_of_drifted_columns") or first["value"].get("drift_share")
            # Вариант 3: данные лежат прямо в первом элементе
            else:
                drift_share = first.get("share_of_drifted_columns") or first.get("drift_share")
    except Exception as e:
        print(f"[evidently] Ошибка при извлечении drift_share: {e}", flush=True)

    if drift_share is not None:
        DRIFT_SHARE.set(float(drift_share))
        print(f"[evidently] drift_share = {drift_share}", flush=True)
    else:
        print("[evidently] Не удалось найти drift_share в структуре", flush=True)

    # --- Безопасное извлечение PSI ---
    try:
        metrics = result.get("metrics", [])
        if metrics:
            first = metrics[0]
            # Ищем drift_by_columns в разных возможных местах
            drift_by_columns = {}
            if "result" in first and isinstance(first["result"], dict):
                drift_by_columns = first["result"].get("drift_by_columns", {})
            elif "value" in first and isinstance(first["value"], dict):
                drift_by_columns = first["value"].get("drift_by_columns", {})
            else:
                drift_by_columns = first.get("drift_by_columns", {})

            for col_name, col_data in drift_by_columns.items():
                score = col_data.get("drift_score")
                if score is not None:
                    PSI_SCORE.labels(feature=col_name).set(float(score))
                    print(f"[evidently] psi {col_name} = {score}", flush=True)
    except Exception as e:
        print(f"[evidently] Ошибка при извлечении PSI: {e}", flush=True)

#TODO: потом в отдельный микросервис
if __name__ == "__main__":
    start_http_server(8000)
    
    while True:
        run_monitoring()
        time.sleep(30)