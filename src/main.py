import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from src.storage.data_manager import DataManager
from src.core.pipeline import forecast_pipeline
from src.schemas.result_schema import PredictResult
from src.core.loading_data import load_data, write_data_to_s3
from src.config.config import RESULT_KEY
from prometheus_fastapi_instrumentator import Instrumentator

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Startup: application is starting up...")
    
    try:
        dm = DataManager()
        app.state.data_manager = dm
    except Exception as e:
        raise RuntimeError(f"Ошибка при инициализации DataManager: {e}") from e
    
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        print("Mlflow успешно инициализирован")
    except Exception as e:
        raise RuntimeError(f"Ошибка при инициализации mlflow: {e}") from e
    
    
    print("Startup: application is ready")
    
    yield
    
app = FastAPI(lifespan=lifespan)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.get("/")
def main():
    return "ok"
    

@app.get("/predict", response_model=PredictResult)
def predict_from_s3(
    file_key: str = Query(..., description="Ключ (путь) к файлу CSV в бакете S3, например: churn_data.csv")
):
    try:
        dm: DataManager = app.state.data_manager
        
        raw_data = load_data(dm = dm, key = file_key)
        result_df = forecast_pipeline(dm = dm, data = raw_data)
        
        return {
            "status": "success",
            "file_processed": file_key,
            "result_filename": RESULT_KEY
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Ошибка в процессе обработки: {str(e)}"
        )
    finally:
        write_data_to_s3(dm = dm, key = RESULT_KEY, data = result_df)

if __name__ == "__main__":
    uvicorn.run(app, host =  "0.0.0.0", port=1111)