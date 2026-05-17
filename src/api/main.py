from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}

@app.get(f"{settings.API_V1_STR}/health")
def health_check():
    return {"status": "healthy"}

@app.post(f"{settings.API_V1_STR}/predict")
def predict():
    # Placeholder for prediction logic
    return {
        "symbol": settings.SYMBOL,
        "direction": "up",
        "confidence": 0.72,
        "regime": "high_volatility"
    }
