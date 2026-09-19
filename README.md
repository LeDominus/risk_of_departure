# Risk of Departure

Сервис для прогнозирования оттока клиентов (customer churn) телеком-оператора.

Проект реализует полный ML-контур: загрузка данных из объектного хранилища (S3 / Yandex Cloud), предобработка, инференс ONNX-модели (логистическая регрессия), постобработка результатов и выдача ответа через REST API. Хранение обученной модели и логирование экспериментов выполняется через **MLflow**, а наблюдение за сервисом — через **Prometheus**.

Работа построена на датасете _Telco Customer Churn_.

---

## Содержание

- [Стек технологий](#стек-технологий)
- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Установка и запуск](#установка-и-запуск)
- [Использование API](#использование-api)
- [Папка данных](#папка-данных)
- [Конфигурация](#конфигурация)
- [Дорожная карта](#дорожная-карта)

---

## Стек технологий

| Область               | Технологии                                                                           |
| --------------------- | ------------------------------------------------------------------------------------ |
| Язык / вычисления     | Python, NumPy, Pandas, scikit-learn                                                  |
| ML-модель             | Logistic Regression (базовая), LightGBM (эксперимент)                                |
| Формат модели         | ONNX, инференс через ONNX Runtime                                                    |
| Хранилище данных      | S3 через `boto3` (Yandex Object Storage, endpoint `https://storage.yandexcloud.net`) |
| Эксперименты / модели | MLflow (сервер через Docker Compose, порт `5000`)                                    |
| API                   | FastAPI + Uvicorn                                                                    |
| Мониторинг            | `prometheus-fastapi-instrumentator` (endpoint `/metrics`)                            |
| Прочее                | `python-dotenv`, Docker Compose                                                      |

---

## Архитектура

Запрос `/predict` проходит по следующему конвейеру:

```
GET /predict?file_key=<key.csv>
        │
        ▼
┌────────────────────────────────────────────────┐
│              FastAPI (main.py)                 │
│  src/core/pipeline.forecast_pipeline()         │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  src/storage/data_manager.DataManager          │
│  Чтение CSV из S3-бакета по file_key           │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  src/core/data_preparation.DataPreparation     │
│  1. Валидация входных колонок (TARGET_COLS)    │
│  2. Предобработка типов (численные фичи)       │
│  3. Label Encoding категориальных признаков    │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  src/core/model.PredictModel                   │
│  Загрузка ONNX-модели из MLflow +              │
│  инференс через ONNX Runtime                   │
└────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────┐
│  postprocess_predictions():                    │
│  customerID / Churn / Probability_of_churn     │
└────────────────────────────────────────────────┘
        │
        ▼
   JSON-ответ
```

Ключевые точки:

- **`PredictModel`** находит актуальный run в эксперименте MLflow `Default` по тегу `model_role = 'logreg_prod'` (последний по `start_time DESC`), скачивает артефакт `model/logistic_regression.onnx` в кэш `tmp/mlflow_cache/` и грузит его через ONNX Runtime.
- **`DataManager`** инкапсулирует работу с S3: инициализацию клиента boto3, health-check бакета (`head_bucket`), чтение и запись объектов.
- **`forecast_pipeline`** — оркестратор: подготовка данных → инференс → постобработка.

---

## Структура проекта

```
RiskOfDeparture/
├── .env                        # Секреты: AWS-ключи, имя бакета, endpoint S3
├── .gitignore
├── docker-compose.yml          # Поднятие MLflow-сервера (порт 5000)
├── requirements.txt            # Зависимости Python
├── test.py                     # Смоук-тест ONNX-модели (проверка входов/выходов)
│
├── data/
│   ├── raw/                    # Исходные данные (CSV / XLS)
│   │   ├── churn_data.csv              # Полный датасет Telco Churn (7043×21, содержит Churn)
│   │   ├── churn_data_new.csv          # Только фичи для инференса (7043×19)
│   │   ├── customer_history.csv        # История клиентов (126789×7)
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.xls
│   └── result/                 # Результаты: графики EDA, важности фич, ONNX-модель
│
├── notebooks/
│   └── eda.ipynb               # EDA, feature engineering, обучение LR и LightGBM,
│                               # экспорт модели в ONNX и логирование в MLflow
│
├── src/
│   ├── main.py                 # Точка входа FastAPI-приложения
│   ├── config/
│   │   └── config.py           # Колонки, параметры модели, пути, переменные окружения
│   ├── core/
│   │   ├── data_preparation.py # Валидация, типы, кодирование, постобработка
│   │   ├── loading_data.py     # Чтение CSV из байтов S3
│   │   ├── model.py            # PredictModel: MLflow + ONNX Runtime инференс
│   │   └── pipeline.py         # forecast_pipeline — оркестратор пайплайна
│   ├── storage/
│   │   ├── __init__.py
│   │   └── data_manager.py     # DataManager: клиент boto3, операции с S3
│   └── utils/
│       └── generate_synth_feature.py   # (в разработке) генерация синтетики
│
├── mlflow_artifacts/           # Артефакты MLflow (смонтированы в контейнер)
├── mlflow_data/                # SQLite-база MLflow
├── mlflow.db                   # Локальная база экспериментов
└── tmp/
    └── mlflow_cache/           # Кэш скачанных ONNX-моделей
```

### Назначение ключевых модулей

- **`src/main.py`** — FastAPI-приложение: `lifespan` инициализирует `DataManager` и настраивает MLflow `tracking_uri`. Эндпоинты: `GET /`, `GET /predict`, `GET /metrics`.
- **`src/config/config.py`** — централизованная конфигурация: список признаков `TARGET_COLS`, числовые признаки `NUM_FEATURES`, параметры модели `MODEL_PARAMS`, загрузка секретов из `.env`.
- **`src/core/data_preparation.py`** — подготовка входных данных к инференсу.
- **`src/core/model.py`** — загрузка и использование ONNX-модели.
- **`src/core/pipeline.py`** — объединение шагов подготовки, прогноза и постобработки.

---

## Установка и запуск

**Требования:** Python 3.8+, Docker (для MLflow), доступ к S3-хранилищу.

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

3. Настройте окружение: создайте файл `.env` на основе шаблона:

   ```env
   AWS_SECRET_KEY_ID=<your_key_id>
   AWS_SECRET_ACCESS_KEY=<your_secret_key>
   BUCKET=<your_bucket>
   S3_ENDPOINT=https://storage.yandexcloud.net
   ```

4. Поднимите MLflow-сервер (порт `5000`):

   ```bash
   docker compose up -d
   ```

5. Запустите API-сервис (порт `1111`):

   ```bash
   python src/main.py
   # или
   uvicorn src.main:app --host 0.0.0.0 --port 1111
   ```

> **Важно:** перед запуском убедитесь, что в MLflow в эксперименте `Default` есть run с тегом `model_role = logreg_prod` и артефактом `model/logistic_regression.onnx`. Пример экспорта модели в ONNX и логирования есть в `notebooks/eda.ipynb`.

---

## Использование API

| Метод | Эндпоинт                  | Описание                                    |
| ----- | ------------------------- | ------------------------------------------- |
| `GET` | `/`                       | Проверка живости сервиса, возвращает `"ok"` |
| `GET` | `/predict?file_key=<key>` | Прогноз оттока по CSV-файлу из S3           |
| `GET` | `/metrics`                | Метрики Prometheus                          |

### Пример запроса прогноза

```bash
curl "http://localhost:1111/predict?file_key=churn_data.csv"
```

`file_key` — путь к CSV-файлу внутри S3-бакета, например `churn_data.csv`.

Пример ответа:

```json
{
  "status": "success",
  "file_processed": "churn_data.csv",
  "data": [
    {
      "customerID": "7590-VHVEG",
      "Churn": 0,
      "Probability_of_churn": 0.13
    }
  ]
}
```

Поля выходных данных:

- `customerID` — уникальный идентификатор клиента;
- `Churn` — прогнозируемый класс (0 — не уйдёт, 1 — уйдёт);
- `Probability_of_churn` — вероятность оттока (округлена до 4 знаков).

## Ошибки обработки возвращаются как `HTTP 400` с деталями ошибки в поле `detail`.

## Папка данных

- **`data/raw/churn_data.csv`** — полный датасет Telco Customer Churn (7043 строк × 21 колонка, содержит целевую переменную `Churn`).
- **`data/raw/churn_data_new.csv`** — набор только с признаками инференса (7043 × 19, без `customerID` и `Churn`) — именно такой формат ожидает модель.
- **`data/raw/customer_history.csv`** — временная история клиентов (126789 × 7): `customerID`, `month`, `MonthlyCharges`, `SupportTickets`, `DataUsage`, `TotalCharges`, `Churn`.
- **`data/result/`** — графики EDA, важности признаков и эталонная ONNX-модель.

---

## Конфигурация

Конфигурация сосредоточена в `src/config/config.py`:

- `TARGET_COLS` — порядок и состав признаков, ожидаемых моделью;
- `NUM_FEATURES` — числовые признаки (`tenure`, `MonthlyCharges`, `SeniorCitizen`, `TotalCharges`);
- `COLS_TO_CONV` — признаки, требующие приведения к числу (`MonthlyCharges`, `TotalCharges`);
- `MODEL_PARAMS` — параметры обучения (зарезервированы для переобучения);
- `MODEL_PATH` — путь к эталонной ONNX-модели в `data/result`;
- секреты (`AWS_SECRET_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `BUCKET`, `S3_ENDPOINT`) загружаются из `.env`.
