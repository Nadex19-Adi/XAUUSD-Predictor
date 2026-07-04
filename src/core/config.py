from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "XAUUSD Predictor"
    API_V1_STR: str = "/api/v1"
    
    # Data Configs
    SYMBOL: str = "GC=F"
    TIMEFRAME: str = "15m"
    HISTORY_PERIOD: str = "60d"
    MASTER_5M_PATH: str = "./data/xauusd_master_5m.csv"
    MASTER_15M_PATH: str = "./data/xauusd_master_15m.csv"
    FEATURES_DATA_PATH: str = "./data/xauusd_features.csv"
    
    # FAISS Vector Index Configs
    FAISS_INDEX_PATH: str = "./data/faiss/market_memory.index"
    FAISS_METADATA_PATH: str = "./data/faiss/metadata.parquet"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Model Configs
    XGB_MODEL_PATH: str = "./models/xgb_model.json"
    
    # MT5 Configs
    MT5_LOGIN: Optional[int] = None
    MT5_PASSWORD: Optional[str] = None
    MT5_SERVER: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
