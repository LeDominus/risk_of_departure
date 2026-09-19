from prometheus_client import Counter, Histogram, Gauge

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of predicted churn probability",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

CHURN_PREDICTED = Counter(
    "churn_predicted_total",
    "Total number of predictions by class",
    ["predicted_class"],  # "0" — не уйдёт, "1" — уйдёт
)

CHURN_RATE = Gauge(
    "churn_predicted_rate",
    "Share of predicted churners in the last batch",
)

BATCH_SIZE = Histogram(
    "predict_batch_size",
    "Input batch size (rows)",
    buckets=[10, 100, 500, 1000, 5000, 10000, 50000],
)

BATCH_ROWS_TOTAL = Counter(
    "predict_rows_total",
    "Total number of rows scored since start",
)

STAGE_LATENCY = Histogram(
    "pipeline_stage_duration_seconds",
    "Latency by pipeline stage",
    ["stage"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

PIPELINE_ERRORS = Counter(
    "pipeline_errors_total",
    "Pipeline errors by stage and type",
    ["stage", "error_type"],
)

MODEL_INFO = Gauge(
    "model_info",
    "Information about the loaded model",
    ["model_name", "model_version", "model_alias"],
)