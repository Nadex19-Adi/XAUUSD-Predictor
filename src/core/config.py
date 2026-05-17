from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "XAUUSD Predictor"
    API_V1_STR: str = "/api/v1"
    
    # Data Configs
    SYMBOL: str = "GC=F"
    TIMEFRAME: str = "5m"
    HISTORY_PERIOD: str = "60d"
    MASTER_DATA_PATH: str = "./data/xauusd_master_5m.csv"
    FEATURES_DATA_PATH: str = "./data/xauusd_features.csv"
    
    # ChromaDB Configs (Shard-Aware)
    CHROMA_PERSIST_DIR: str = "./data/chromadb"
    SHARD_COLLECTIONS: List[str] = ["gold_legacy", "gold_mid", "gold_recent"]
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Model Configs
    XGB_MODEL_PATH: str = "./models/xgb_model.json"
    XGB_MASTER_MODEL_PATH: str = "./models/xauusd_model.bst"
    
    # MT5 Configs
    MT5_LOGIN: Optional[int] = None
    MT5_PASSWORD: Optional[str] = None
    MT5_SERVER: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
