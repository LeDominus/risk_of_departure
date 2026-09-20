import pandas as pd
from io import BytesIO


def buffer_data(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    return buf