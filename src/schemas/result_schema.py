from pydantic import BaseModel

class PredictResult(BaseModel):
    status: str
    file_processed: str
    result_filename: str