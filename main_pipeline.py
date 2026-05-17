from src.data.ingestion import fetch_historical_data
from src.features.technical import add_technical_indicators
from src.models.rag_model import RAGMarketMemory
from src.models.xgb_model import XGBPredictor
import pandas as pd
import os

def run_training_pipeline():
    print("1. Fetching Data...")
    df = fetch_historical_data()
    
    print(f"2. Adding Technical Features... (Initial shape: {df.shape})")
    df = add_technical_indicators(df)
    
    print("3. Initializing RAG Market Memory...")
    rag = RAGMarketMemory()
    
    # We rebuild the memory database with historical data
    # In a perfect setup, we wipe the DB and rebuild to prevent duplication
    print("   -> Embedding chunks into ChromaDB (This may take a moment)...")
    rag.build_memory(df)
    
    print("4. Augmenting Features with RAG Similarities...")
    rag_features = []
    
    # For speed in this pipeline demo, we will only query the last 1000 rows.
    # In full production, you query all rows (which takes time).
    subset_df = df.tail(1000) 
    
    for idx, row in subset_df.iterrows():
        # strict timestamp filtering prevents look-ahead
        sim_data = rag.query_state(row, str(idx), top_k=5)
        rag_features.append(sim_data)
        
    rag_df = pd.DataFrame(rag_features, index=subset_df.index)
    final_df = subset_df.join(rag_df).dropna()
    
    print(f"5. Training XGBoost Model with Walk-Forward Validation... (Training rows: {len(final_df)})")
    xgb_model = XGBPredictor()
    accuracy = xgb_model.train_walk_forward(final_df, n_splits=5)
    
    print("6. Saving Model...")
    os.makedirs("./data/models", exist_ok=True)
    xgb_model.save()
    
    print(f"Pipeline Complete! Final Walk-Forward Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    run_training_pipeline()
