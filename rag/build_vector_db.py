import pandas as pd
import os
import gc
import argparse
from rag.rag_engine import MarketRAG


def build_db(csv_path: str = "data/xauusd_features.csv"):
    """
    Builds the FAISS vector index from the full feature CSV.
    
    Replaces the old ChromaDB sharded build process.
    Old: 3 shards × ~560K rows each → 9.7 GB on disk
    New: 1 unified FAISS IVF-PQ index → ~150 MB on disk
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Building FAISS Market Memory Index from {csv_path}...")
    
    rag = MarketRAG()
    
    # Load the full dataset
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Validate required columns
    required = ['next_5m_direction', 'next_return']
    for col in required:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in dataset.")
            return
    
    # Drop rows with NaN in critical columns
    initial_len = len(df)
    df = df.dropna(subset=required)
    if len(df) < initial_len:
        print(f"  Dropped {initial_len - len(df)} rows with NaN targets. Remaining: {len(df)}")
    
    # Build the FAISS index (embedding + indexing + saving)
    rag.build_index(df, batch_size=5000)
    
    # Cleanup
    del df
    gc.collect()
    
    print(f"\n[DONE] FAISS index build complete.")
    print(f"  Index: data/faiss/market_memory.index")
    print(f"  Metadata: data/faiss/metadata.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS Vector Index for Market Memory")
    parser.add_argument("--csv", type=str, default="data/xauusd_features.csv",
                        help="Path to the feature CSV file")
    args = parser.parse_args()
    
    build_db(csv_path=args.csv)
