import numpy as np
from src.monitoring.metrics import (
    BATCH_ROWS_TOTAL, BATCH_SIZE,
    CHURN_PREDICTED, CHURN_PROBABILITY, CHURN_RATE
)

def emit_batch_metrics(probs: np.ndarray) -> None:
    """Считает и регистрирует метрики батча."""
    n = len(probs)
    BATCH_SIZE.observe(n)
    BATCH_ROWS_TOTAL.inc(n)

    for p in probs:
        CHURN_PROBABILITY.observe(float(p))

    preds = (probs >= 0.5).astype(int)
    n_positive = int(preds.sum())
    n_negative = n - n_positive
    CHURN_PREDICTED.labels(predicted_class="1").inc(n_positive)
    CHURN_PREDICTED.labels(predicted_class="0").inc(n_negative)

    if n > 0:
        CHURN_RATE.set(n_positive / n)