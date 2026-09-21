# Risk of Departure

Сервис прогнозирования оттока клиентов (**customer churn**) телеком-оператора на данных
_Telco Customer Churn_. Проект представляет собой **полный ML-контур от обучения модели
до производственной эксплуатации (MLOps)**: EDA и обучение моделей в ноутбуках, экспорт
в ONNX, версионирование и хранение моделей через **MLflow**, батч-инференс через REST API
на **FastAPI**, логирование прогнозов в объектное хранилище, аналитика поверх прогнозов
через **DuckDB**, отслеживание дрейфа данных через **Evidently** и визуализация всего
стека мониторинга в **Prometheus + Grafana**.

---

## Содержание

- [Решаемые задачи](#решаемые-задачи)
- [Стек технологий](#стек-технологий)
- [Архитектура](#архитектура)
- [ML-контур: обучение и экспорт модели](#ml-контур-обучение-и-экспорт-модели)
- [Инференс-контур: REST API](#инференс-контур-rest-api)
- [Контур логирования прогнозов](#контур-логирования-прогнозов)
- [Мониторинг и наблюдаемость](#мониторинг-и-наблюдаемость)
- [Структура проекта](#структура-проекта)
- [Установка и запуск](#установка-и-запуск)
- [Использование API](#использование-api)
- [Данные](#данные)
- [Конфигурация](#конфигурация)
- [Дорожная карта](#дорожная-карта)

---

## Решаемые задачи

1. **Прогнозирование оттока** — по признакам клиента (тариф, услуги, длительность,
   платежи) предсказать класс оттока (`Churn: 0/1`) и вероятность ухода
   (`Probability_of_churn`).
2. **Полный цикл ML-модели** — EDA, обучение нескольких моделей (базовая
   логистическая регрессия + эксперимент LightGBM), оценка качества, экспорт в
   переносимый формат **ONNX**, версионирование и хранение в **MLflow**.
3. **Батч-инференс из объектного хранилища** — чтение CSV напрямую из **S3**, единый
   пайплайн подготовки данных → инференс → постобработка, запись результата обратно в S3.
4. **Версионирование моделей и экспериментов** — воспроизводимость run-ов,
   фиксация метрик и артефактов через **MLflow Tracking Server**.
5. **Наблюдаемость сервиса** — метрики производительности API (RPS, задержки, коды
   ответов) и ресурсы процесса (CPU/RAM) через **Prometheus**.
6. **ML-мониторинг качества данных** — детекция **дрейфа данных (data drift)** и расчёт
   **PSI** по фичам через **Evidently**, сравнение текущего распределения с референсным.
7. **Накопление истории прогнозов** — логирование каждого батча в **Parquet** с
   hive-партиционированием в S3 для последующего анализа и обучения на отложенных
   метках (`delayed labels`).
8. **Аналитика поверх прогнозов** — `DuckDB + httpfs` читает Parquet прямо из S3,
   дедуплицирует по `prediction_id` (скользящее окно) для кормёжки Evidently.

---

## Стек технологий

| Область                  | Технологии                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| Язык / вычисления        | Python, NumPy, Pandas, scikit-learn                                                                    |
| ML-модели                | Logistic Regression (базовая, production), LightGBM (эксперимент)                                      |
| Формат / инференс модели | ONNX, экспорт через `skl2onnx`, инференс через ONNX Runtime (`onnxruntime`)                            |
| Работа с дисбалансом     | `class_weight='balanced'`, библиотека `imblearn` (SMOTE/undersampling)                                 |
| Хранилище данных         | S3 через `boto3` (Yandex Object Storage, region `ru-central1`, path-style addressing)                  |
| Эксперименты / модели    | MLflow Tracking Server (Docker Compose, порт `5000`, backend — SQLite, артефакты — смонтированный том) |
| API                      | FastAPI + Uvicorn (порт `1111`)                                                                        |
| Мониторинг API           | `prometheus-fastapi-instrumentator` + кастомные метрики `prometheus_client` (эндпоинт `/metrics`)      |
| ML-мониторинг (дрейф)    | Evidently (`DataDriftPreset`: доля дрейфующих фич + PSI), отдельный экспортёр-сервис на порту `8000`   |
| Аналитика прогнозов      | DuckDB + расширение `httpfs` (чтение Parquet с S3, hive-partitioning, window-дедупликация)             |
| Визуализация / дашборды  | Prometheus (порт `9091`), Grafana (порт `3001`, provisioning-дашборды: API, ML-метрики, Evidently)     |
| Промышленные практики    | Data-классы Pydantic, Docker Compose, `python-dotenv`, централизованный `config.py`                    |
| Анализ / обучение        | Jupyter (EDA, feature engineering, обучение, экспорт и логирование моделей), Matplotlib, Seaborn       |

---

## Архитектура

```
                      ┌─────────────────────────────────────────────────────────────┐
                      │                   Обучение (notebooks/eda.ipynb)             │
                      │ EDA → препроцессинг → LR (Production) / LightGBM (эксп.)    │
                      │ оценка (Acc/F1/ROC-AUC) → skl2onnx → ONNX → MLflow (run)   │
                      └─────────────────────────────────────────────────────────────┘
                                        │ model/logistic_regression.onnx (RUN_ID)
                                        ▼
 GET /predict?file_key=<key.csv>  ┌────────────────────────────────────────────────┐
        │                          │  FastAPI (src/main.py)                         │
        ▼                          │  forecast_pipeline()                            │
┌───────────────────────────────┐  └────────────────────────────────────────────────┘
│ DataManager (boto3, S3)       │   ├─ DataPreparation: валидация колонок (TARGET_COLS),
│ чтение CSV из бакета по ключу │   │   приведение типов, Label Encoding
└───────────────────────────────┘   ├─ PredictModel: скачивание ONNX из MLflow в кэш
        │                           │   tmp/mlflow_cache → инференс ONNX Runtime
        ▼                           ├─ postprocess_predictions(): customerID/Churn/prob
┌───────────────────────────────┐   └─ запись результата forecast_result.csv обратно в S3
│  JSON-ответ (PredictResult)   │
└───────────────────────────────┘
        │
        ▼       Пост-обработка каждого батча (best-effort, не блокирует ответ)
┌────────────────────────────────────────────────────────────────────────────────────┐
│ append_pred_log_to_s3(): фичи + churn + probability + prediction_id → Parquet,      │
│ hive-партиционирование по prediction_date → S3 (PREDICTIONS_PREFIX)                │
└────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│ Evidently-monitor (:8000) — DuckDB httpfs читает Parquet из S3 (скользящее окно,    │
│ дедуп по prediction_id) → DataDriftPreset (drift_share, PSI по фичам)              │
│            │                                                                       │
│            ▼                                                                       │
│      Prometheus (:9091) ← /metrics (FastAPI) →  Grafana (:3001) → дашборды          │
└────────────────────────────────────────────────────────────────────────────────────┘
```

Ключевые точки:

- **`PredictModel`** — загружает ONNX-модель конкретного `RUN_ID` из MLflow в локальный
  кэш `tmp/mlflow_cache/`, создаёт `onnxruntime.InferenceSession`, выполняет инференс и
  возвращает метки класса и вероятности (обрабатывает разные форматы выхода ONNX:
  `dict`, `ndarray`, `list`).
- **`DataManager`** — инкапсулирует S3: инициализация клиента boto3 (path-style,
  ru-central1), health-check бакета (`head_bucket`, разбор кодов 404/403), чтение/запись.
- **`DataPreparation`** — единый препроцессинг инференса: отбрасывание лишних колонок,
  приведение строковых чисел к числовым (`COLS_TO_CONV`), `' '`→`NaN`→0 и `LabelEncoder`
  для категориальных фич. Порядок фич строго соответствует `TARGET_COLS` (важно для
  ONNX-инференса по фиксированному входному тензору).
- **`forecast_pipeline`** — оркестратор: подготовка → инференс → постобработка, с замером
  латентности по стадиям и логированием метрик/предсказаний даже при сбое (`finally`).
- **`DuckDBManager`** — читает Parquet-прогнозы прямо из S3 через `httpfs`, настраивает
  `s3_endpoint/keys/region`, применяет window-функцию `row_number()` для дедупликации по
  `prediction_id` и отдаёт «свежую» выборку.

---

## ML-контур: обучение и экспорт модели

Весь этап обучения сосредоточен в `notebooks/eda.ipynb` и повторяет стандартную
исследовательскую процедуру для табулярной бинарной классификации:

1. **EDA/GVA** — загрузка `Telco Customer Churn`, анализ распределений, пропусков,
   корреляций (`Matplotlib`/`Seaborn`), сохранение графиков в `data/result/`.
2. **Препроцессинг** — отбор признаков, приведение типов, Label Encoding категориальных
   переменных.
3. **Сплит** — `train_test_split` (test_size = 0.15, `random_state = 42`).
4. **Базовая модель: Logistic Regression** — `class_weight='balanced'` для компенсации
   дисбаланса классов (доля оттока ~26%), `n_jobs=-1`, `max_iter=100`.
5. **Эксперимент: LightGBM** — `LGBMClassifier(n_estimators=300, reg_alpha=1,
reg_lambda=1)` — сравнивается с базовой моделью по качеству.
6. **Оценка** — `Accuracy`, `F1`, `ROC-AUC`, `classification_report`, `confusion_matrix`,
   ROC-кривые, визуализация важности признаков (для LR — топ-20 коэффициентов; для
   LightGBM — feature importance). Графики сохраняются в `data/result/`.
7. **Экспорт в ONNX** — базовая логистическая регрессия конвертируется через
   `skl2onnx` (`to_onnx`, на максимальном поддерживаемом opset) в
   `logistic_regression.onnx` и сохраняется в `data/result/`.
8. **Версионирование в MLflow** — модель и метрики логируются в Tracking Server
   (см. `RUN_ID` в `src/config/config.py`). Онлайн-сервис тянет именно артефакт
   `model/logistic_regression.onnx` этого run-а. Для выгрузки модели без запуска сервиса
   есть скрипт `src/scripts/download_model.py`.

> Дополнительно: для борьбы с дисбалансом в зависимостях присутствует `imblearn`
> (SMOTE / undersampling) — доступно для переобучения.

---

## Инференс-контур: REST API

Запускается процессом FastAPI/Uvicorn (порт `1111`), все тяжёлые инициализации
выполняются в `lifespan` (подготовка `DataManager`, настройка `tracking_uri` MLflow).

| Этап                    | Модуль                                      | Действие                                                     |
| ----------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| Чтение данных           | `src/core/loading_data.load_data`           | `GET` объекта из S3 по `file_key`, парсинг CSV               |
| Подготовка данных       | `src/core/data_preparation.DataPreparation` | валидация колонок, типы, Label Encoding                      |
| Инференс                | `src/core/model.PredictModel`               | скачивание ONNX из MLflow, `onnxruntime` predict             |
| Постобработка           | `DataPreparation.postprocess_predictions`   | `customerID / Churn / Probability_of_churn` (prob округлена) |
| Ответ                   | `src/schemas/result_schema.PredictResult`   | JSON `{status, file_processed, result_filename}`             |
| Персистенция результата | `src/core/loading_data.write_data_to_s3`    | запись `forecast_result.csv` обратно в S3                    |

Ключевые особенности:

- **Пайплайн не блокирует ответ логированием**: запись предсказаний в S3 и отправка
  метрик выполняются best-effort (ошибки логируются, но не пробрасываются клиенту).
- **Обработка ошибок**: любая ошибка пайплайна превращается в `HTTP 400` с деталями в
  `detail`, а счётчик `pipeline_errors_total` инкрементируется.
- **Схема ответа фиксирована** Pydantic-моделью `PredictResult`. Полезная нагрузка с
  самими предсказаниями сохраняется в S3 (файл `forecast_result.csv`).

---

## Контур логирования прогнозов

Каждый прогнозируемый батч дополнительно пишется в S3 в виде **Parquet**-файлов с
hive-партиционированием (модуль `src/storage/prediction_writer.py`,
`append_pred_log_to_s3`):

- путь: `{PREDICTIONS_PREFIX}/prediction_date={YYYY-MM-DD}/part-{HHMMSS_ff}.parquet`;
- состав: признаки (как подают в модель) + служебные (`customer_id`), а также
  `prediction_id` (UUID на строку), `prediction_date`, `model_version`, `churn`
  (Int8) и `probability` (float32);
- сериализация через `buffer_data` (BytesIO → `df.to_parquet`, engine `pyarrow`).

Дублирующий локальный канал — `src/utils/log_entries.py` пишет JSONL в
`data/logs/prediction_logs.jsonl` (используется для быстрой отладки; основной канал —
S3).

`prediction_id` служит уникальным ключом строки, что позволяет **дедуплицировать**
повторные/частичные записи при чтении (см. DuckDB window-функцию).

---

## Мониторинг и наблюдаемость

Стек мониторинга поднимается через Docker Compose: **Prometheus → Grafana**, плюс
отдельный сервис **Evidently-monitor**.

### Метрики HTTP-сервиса (`src/monitoring/metrics.py`, эндпоинт `/metrics`)

| Метрика                           | Тип       | Описание                                                    |
| --------------------------------- | --------- | ----------------------------------------------------------- |
| `churn_probability`               | Histogram | Распределение предсказанной вероятности оттока              |
| `churn_predicted_total`           | Counter   | Всего предсказаний по классу (`predicted_class`)            |
| `churn_predicted_rate`            | Gauge     | Доля «ушедших» в последнем батче                            |
| `predict_batch_size`              | Histogram | Размер входного батча                                       |
| `predict_rows_total`              | Counter   | Всего строк, прошедших скоринг                              |
| `pipeline_stage_duration_seconds` | Histogram | Латентность по стадиям (read/prepare/inference/postprocess) |
| `pipeline_errors_total`           | Counter   | Ошибки по стадии и типу исключения                          |
| `model_info`                      | Gauge     | Информация о загруженной модели (name/version/alias)        |

Стандартные метрики FastAPI (RPS, латентность, коды 2xx/5xx, память/CPU процесса)
автоматически добавляются через `prometheus-fastapi-instrumentator`.

### ML-мониторинг через Evidently (`src/monitoring/evidently_monitoring.py`, порт `8000`)

- Каждые `INTERVAL_SECONDS` (30 с) забирает через `DuckDBManager` последние
  `WINDOW_ROWS` (1000) прогнозов из S3 (hive-partitioning + дедупликация по
  `prediction_id`).
- Сравнивает со скользящим окном с **референсным распределением**
  (`data/raw/churn_data_new.csv`, фичи из `FEATURE_COLS`).
- Запускает `Evidently Report([DataDriftPreset])` и публикует в Prometheus:
  - `evidently_drift_share` — доля дрейфующих фич;
  - `evidently_psi_score{feature}` — PSI по каждой фиче;
  - `evidently_current_rows` — размер текущего окна;
  - `evidently_s3_query_seconds` — латентность запросов к S3 через DuckDB.
- Если накоплено меньше `MIN_SAMPLE_SIZE` (300) строк — расчёт пропускается.

### Инфраструктура мониторинга (docker-compose)

| Сервис     | Порт   | Роль                                                                 |
| ---------- | ------ | -------------------------------------------------------------------- |
| MLflow     | `5000` | Tracking Server (SQLite + артефакты)                                 |
| Prometheus | `9091` | Сбор метрик (scrape fastapi `host.docker.internal:1111` и Evidently) |
| Grafana    | `3001` | Дашборды (provisioning: API, ML-метрики, Evidently)                  |
| Evidently  | `8000` | Дрейф-мониторинг и экспорт метрик в Prometheus                       |

---

## Структура проекта

```
RiskOfDeparture/
├── .env                          # Секреты: AWS-ключи, имя бакета, endpoint S3
├── .gitignore
├── docker-compose.yml            # mlflow, prometheus, grafana, evidently-monitor
├── prometheus.yml                # Конфиг scrape-таргетов Prometheus
├── requirements.txt              # Зависимости Python
│
├── data/
│   ├── raw/                      # Исходные данные (CSV / XLS)
│   │   ├── churn_data.csv               # Полный датасет Telco Churn (7043×21, с Churn)
│   │   ├── churn_data_new.csv           # Только фичи для инференса (7043×19)
│   │   ├── customer_history.csv         # История клиентов (126789×7) — future feature store
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.xls
│   ├── logs/                     # Локальный JSONL-лог предсказаний
│   └── result/                   # Графики EDA, важности фич, эталонный ONNX
│
├── grafana/
│   └── provisioning/             # Дашборды и datasource (FastAPI, ML, Evidently)
│
├── notebooks/
│   ├── eda.ipynb                 # EDA, обучение LR/LightGBM, ONNX-экспорт, MLflow
│   └── test.ipynb                # Проверка модели/пайплайна
│
├── mlflow_artifacts/             # Артефакты MLflow (смонтированы в контейнер)
├── mlflow_data/                  # SQLite-база MLflow backend-store
│
├── src/
│   ├── main.py                   # Точка входа FastAPI (lifespan, эндпоинты)
│   ├── config/
│   │   ├── config.py             # Колонки, параметры, RUN_ID, пути, секреты
│   │   └── monitoring_config.py  # Параметры мониторинга (окно, порог, пути)
│   ├── core/
│   │   ├── data_preparation.py   # DataPreparation: валидация/типы/кодирование
│   │   ├── loading_data.py       # read CSV из байтов S3, write result в S3
│   │   ├── model.py              # PredictModel: MLflow + ONNX Runtime
│   │   └── pipeline.py           # forecast_pipeline — оркестратор
│   ├── monitoring/
│   │   ├── evidently_monitoring.py  # Дрейф/PSI через Evidently + Prometheus
│   │   ├── metrics.py               # Определения Prometheus-метрик
│   │   ├── metrics_registry.py      # emit_batch_metrics для батча
│   │   └── Dockerfile               # Контейнер Evidently-monitor
│   ├── schemas/
│   │   ├── feature_schema.py     # FEATURE_COLS и SERVICE_COLS
│   │   └── result_schema.py      # Pydantic PredictResult
│   ├── scripts/
│   │   └── download_model.py     # Выгрузка ONNX из MLflow в кэш
│   ├── storage/
│   │   ├── data_manager.py       # DataManager: boto3/S3 клиент
│   │   ├── duckdb_manager.py     # DuckDB + httpfs чтение Parquet с S3
│   │   └── prediction_writer.py  # Логирование прогнозов в Parquet (S3)
│   └── utils/
│       ├── buffer.py             # BufferedIO parquet-сериализация
│       └── log_entries.py        # JSONL-лог предсказаний (локально)
│
├── tmp/
│   └── mlflow_cache/             # Кэш скачанных ONNX-моделей
└── docs/
    └── what_to_do.md             # Технический бэклог / дорожная карта
```

### Назначение ключевых модулей

- **`src/main.py`** — FastAPI-приложение: `lifespan` инициализирует `DataManager` и
  `tracking_uri` MLflow; эндпоинты `GET /`, `GET /predict`, `GET /metrics`.
- **`src/core/pipeline.forecast_pipeline`** — ядро ML-контура: подготовка → инференс →
  постобработка + неблокирующий сбор метрик и логирования прогнозов.
- **`src/storage/duckdb_manager`** — аналитический слой: читает Parquet из S3 напрямую
  (без скачивания), дедуплицирует по `prediction_id`.
- **`src/monitoring/evidently_monitoring`** — дрейф-мониторинг, работает как
  самодостаточный процесс-экспортёр метрик.

---

## Установка и запуск

**Требования:** Python 3.8+ (рекомендуется 3.11), Docker + Docker Compose, доступ к
S3-хранилищу (Yandex Object Storage).

1. Клонируйте репозиторий и создайте виртуальное окружение:

   ```bash
   git clone <repo-url>
   cd RiskOfDeparture
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```

2. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

3. Настройте окружение: создайте `.env` на основе шаблона:

   ```env
   AWS_SECRET_KEY_ID=<your_key_id>
   AWS_SECRET_ACCESS_KEY=<your_secret_key>
   BUCKET=<your_bucket>
   S3_ENDPOINT=https://storage.yandexcloud.net
   S3_REGION=ru-central1
   S3_PREDICTIONS_PREFIX=risk-of-departure/predictions
   ```

4. Поднимите инфраструктуру (MLflow, Prometheus, Grafana, Evidently):

   ```bash
   docker compose up -d
   ```

   - MLflow: `http://localhost:5000`;
   - Prometheus: `http://localhost:9091`;
   - Grafana: `http://localhost:3001` (admin/admin);
   - Evidently-monitor: `http://localhost:8000`.

5. Запустите API-сервис (порт `1111`):

   ```bash
   python src/main.py
   # или
   uvicorn src.main:app --host 0.0.0.0 --port 1111
   ```

> **Важно:** перед запуском убедитесь, что в MLflow существует run с идентификатором,
> заданным в `RUN_ID` (см. `src/config/config.py`), и артефактом
> `model/logistic_regression.onnx`. Пример обучения, экспорта в ONNX и логирования в
> MLflow есть в `notebooks/eda.ipynb`; без запущенного сервера MLflow возникает ошибка
> загрузки модели.
> | Prometheus | `9091` | Сбор метрик (scrape fastapi `host.docker.internal:1111` и Evidently) |
> | Grafana | `3001` | Дашборды (provisioning: API, ML-метрики, Evidently) |

## | Evidently | `8000` | Дрейф-мониторинг и экспорт метрик в Prometheus |

## Использование API

| Метод | Эндпоинт                  | Описание                          |
| ----- | ------------------------- | --------------------------------- |
| `GET` | `/`                       | Health-check, возвращает `"ok"`   |
| `GET` | `/predict?file_key=<key>` | Прогноз оттока по CSV-файлу из S3 |
| `GET` | `/metrics`                | Метрики Prometheus                |

### Пример запроса прогноза

```bash
curl "http://localhost:1111/predict?file_key=churn_data.csv"
```

`file_key` — путь к CSV-файлу внутри S3-бакета, например `churn_data.csv`.

Пример ответа (`PredictResult`, response_model):

```json
{
  "status": "success",
  "file_processed": "churn_data.csv",
  "result_filename": "forecast_result.csv"
}
```

Полные предсказания (построчно: `customerID`, `Churn`, `Probability_of_churn`) пишутся в
файл `forecast_result.csv` в S3 (ключ `RESULT_KEY`). Поля, которые возвращаются в
файле:

- `customerID` — уникальный идентификатор клиента;
- `Churn` — прогнозируемый класс (0 — не уйдёт, 1 — уйдёт);
- `Probability_of_churn` — вероятность оттока (округлена до 4 знаков).

Ошибки обработки возвращаются как `HTTP 400` с деталями в поле `detail`.

> Файл `data/result/forecast_result.csv` — пример локального результата инференса.

---

## Данные

- **`data/raw/churn_data.csv`** — полный датасет Telco Customer Churn (7043 × 21,
  содержит целевую переменную `Churn`).
- **`data/raw/churn_data_new.csv`** — набор только с признаками инференса (7043 × 19,
  без `customerID`/`Churn`) — именно такой формат использует Evidently как референс.
- **`data/raw/customer_history.csv`** — временная история клиентов
  (126789 × 7): `customerID`, `month`, `MonthlyCharges`, `SupportTickets`, `DataUsage`,
  `TotalCharges`, `Churn` — потенциальная основа для Feature Store и real-time фич.
- **`data/raw/churn_data_{1..5}.csv`** — разбитый на партиции датасет для тестов
  инференса.
- **`data/result/`** — графики EDA, важности признаков (LR/LightGBM), эталонная
  ONNX-модель и пример результата прогноза.
- **`data/logs/prediction_logs.jsonl`** — локальный JSONL-лог предсказаний (для
  отладки; основной канал — S3/Parquet).

---

## Конфигурация

### `src/config/config.py`

- `TARGET_COLS` / `FEATURE_COLS` — состав и порядок признаков, ожидаемый моделью;
- `NUM_FEATURES` — числовые признаки (`tenure`, `MonthlyCharges`, `SeniorCitizen`,
  `TotalCharges`);
- `COLS_TO_CONV` — признаки, требующие приведения к числу (`MonthlyCharges`,
  `TotalCharges` — строки с пропусками `' '`);
- `MODEL_PARAMS` — параметры обучения (зарезервированы для переобучения);
- `RUN_ID` — идентификатор run в MLflow, из которого грузится модель;
- `MLFLOW_TRACKING_URI` — адрес MLflow (по умолчанию `http://localhost:5000`);
- `MLFLOW_MODEL_NAME`, `MLFLOW_CACHE_DIR` — имя артефакта и путь кэша;
- `MODEL_PATH` — путь к эталонной ONNX-модели в `data/result`;
- `RESULT_KEY`, `PREDICTIONS_PREFIX` — ключи записи результата и логов прогнозов в S3;
- секреты (`AWS_SECRET_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BUCKET`, `S3_ENDPOINT`,
  `S3_REGION`) загружаются из `.env` через `python-dotenv`.

### `src/config/monitoring_config.py`

- `LOG_PATH`, `PREDICTION_LOG_PATH` — локальный JSONL-лог предсказаний;
- `REFERENCE_PATH` — референсный датасет для Evidently;
- `INTERVAL_SECONDS` (30) — период расчёта дрейфа;
- `WINDOW_ROWS` (1000) — размер скользящего окна;
- `MIN_SAMPLE_SIZE` (300) — минимальный размер выборки для расчёта.

