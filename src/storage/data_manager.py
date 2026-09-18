from boto3.session import Session
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from src.config.config import AWS_SECRET_ACCESS_KEY, AWS_SECRET_KEY_ID, BUCKET, S3_ENDPOINT

class DataManager:
    def __init__(self):
        self.aws_secret_key_id = AWS_SECRET_KEY_ID
        self.aws_secret_access_key = AWS_SECRET_ACCESS_KEY
        self.bucket = BUCKET
        self.s3_endpoint = S3_ENDPOINT
        
        self.client = None
        
        self._init_client()
        self._health_check()
    
    
    def _init_client(self) -> None:
        """Инициализация клиента"""
        try:
            session = Session()
            
            s3_config = Config(
                s3={'addressing_style': 'path'}
            )
            
            client = session.client(
                service_name = "s3",
                region_name='ru-central1',
                endpoint_url = self.s3_endpoint,
                aws_access_key_id = self.aws_secret_key_id,
                aws_secret_access_key = self.aws_secret_access_key,
                config=s3_config
            )
            self.client = client
        except Exception as e:
            raise RuntimeError(f"[DataManager] Ошибка при инициализации клиента boto3: {e}") from e
    
    def _health_check(self) -> None:
        """Проверка жизнеспособности бакета"""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            print(f"Бакет '{self.bucket}' доступен и работает корректно")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"Бакет '{self.bucket}' не существует")
            elif error_code == '403':
                print(f"Бакет '{self.bucket}' существует, но у вас нет прав доступа (Доступ запрещен)")
            else:
                print(f"Ошибка при обращении к бакету: {e}")
    
    def read_data(self, key: str) -> pd.DataFrame:
        """Чтение данных из хранилища S3"""
        if key == None:
            raise ValueError(f"[DataManager] Ключ не может быть None")
        try:
            response = self.client.get_object(Bucket = self.bucket, Key = key)
            file = response['Body'].read()
            return file
        except Exception as e:
            raise ValueError(f"[DataManager] Ошибка при получении файла: {e}") from e
        
    def write_data(self, data: pd.DataFrame, key: str) -> None:
        """Запись данных в хранилище S3"""
        if key == None:
            raise ValueError(f"[DataManager] Ключ не может быть None") 
        try:
            self.client.put_object(
                Bucket = self.bucket,
                Key = key,
                Body = data,
            )
        except Exception as e:
            raise ValueError(f"[DataManager] Ошибка при отправке данных: {e}") from e