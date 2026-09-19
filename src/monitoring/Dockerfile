FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    evidently \
    pandas \
    prometheus-client

COPY evidently_monitoring.py .

CMD ["python", "evidently_monitoring.py"]