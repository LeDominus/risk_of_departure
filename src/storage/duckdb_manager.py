import time
import pandas as pd
import duckdb
from src.config.config import (
    BUCKET, S3_ENDPOINT, S3_REGION,
    AWS_SECRET_ACCESS_KEY, AWS_SECRET_KEY_ID, PREDICTIONS_PREFIX
)

class DuckDBManager:
    def __init__(self):
        self.s3_endpoint = S3_ENDPOINT
        self.s3_access_key_id = AWS_SECRET_KEY_ID
        self.s3_secret_access_key = AWS_SECRET_ACCESS_KEY
        self.s3_use_ssl = True
        self.s3_url_style = 'path'
        self.s3_region = S3_REGION
        
        self.connection: duckdb.DuckDBPyConnection | None = None
        self._connect()
        
    
    def _connect(self) -> None:
        """Создаёт соединение и настраивает httpfs под S3."""
        endpoint = self.s3_endpoint.replace("https://", "").replace("http://", "")

        try:
            con = duckdb.connect()
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.execute(f"SET s3_endpoint='{endpoint}';")
            con.execute(f"SET s3_access_key_id='{self.s3_access_key_id}';")
            con.execute(f"SET s3_secret_access_key='{self.s3_secret_access_key}';")
            con.execute(f"SET s3_use_ssl={'true' if self.s3_use_ssl else 'false'};")
            con.execute(f"SET s3_url_style='{self.s3_url_style}';")
            con.execute(f"SET s3_region='{self.s3_region}';")

            self.connection = con
        except Exception as e:
            raise RuntimeError(f"[DuckDBManager] ошибка подключения: {e}") from e

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            
    def __enter__(self) -> "DuckDBManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
    
    def _glob(self, prefix: str) -> str:
        return f"s3://{BUCKET}/{prefix}/**/*.parquet"
    
    def parse_logs(self, limit: int = 1000) -> pd.DataFrame:
        """Чтение последних N предсказаний из S3 с дедупликацией по prediction_id"""
        if self.connection is None:
            raise RuntimeError("[DuckDBManager] соединение не открыто")

        glob = self._glob(PREDICTIONS_PREFIX)

        sql = f"""
            WITH raw AS (
                SELECT *
                FROM read_parquet('{glob}', hive_partitioning=true)
            ),
            dedup AS (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY prediction_id
                           ORDER BY prediction_date DESC
                       ) AS rn
                FROM raw
            )
            SELECT * FROM dedup
            WHERE rn = 1
            ORDER BY prediction_date DESC
            LIMIT {limit}
        """

        t0 = time.perf_counter()
        df = self.connection.execute(sql).df()
        elapsed = time.perf_counter() - t0

        print(
            f"[DuckDBManager] parse_logs: {len(df)} rows in {elapsed:.3f}s",
            flush=True,
        )
        return df