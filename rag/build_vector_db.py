import pandas as pd
import os
import gc
import chromadb
from rag.rag_engine import MarketRAG

import argparse

def build_db(csv_path: str = "data/xauusd_features.csv", target_shard: str = "all"):
    """
    Builds the vector database by sharding 1.7M records.
    Can build specific shards if requested.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    db_dir = "./data/chromadb"
    print(f"Targeting ChromaDB at {db_dir}...")
    
    rag = MarketRAG()
    
    print(f"Loading dataset for shard: {target_shard}...")
    
    # Sharding Logic
    shards_config = {
        "legacy": (0, 560000, "gold_legacy"),
        "mid": (560000, 560000, "gold_mid"),
        "recent": (1120000, 600000, "gold_recent") # Estimates
    }

    if target_shard == "all":
        to_process = ["legacy", "mid", "recent"]
        # For 'all', we might still want to load chunk by chunk or shard by shard
    else:
        to_process = [target_shard]

    for shard_key in to_process:
        start, nrows, col_name = shards_config[shard_key]
        
        # Load ONLY the required shard
        print(f"Reading rows {start} to {start + nrows if nrows else 'end'} from {csv_path}...")
        
        # We need to handle header correctly if skipping
        if start == 0:
            shard_df = pd.read_csv(csv_path, index_col=0, parse_dates=True, nrows=nrows)
        else:
            # Read header first to get column names
            header_df = pd.read_csv(csv_path, index_col=0, parse_dates=True, nrows=0)
            shard_df = pd.read_csv(csv_path, index_col=0, parse_dates=True, skiprows=range(1, start + 1), nrows=nrows)
            shard_df.columns = header_df.columns
        
        print(f"\n>> Processing Shard: {col_name} ({len(shard_df)} rows)...")
        rag.add_to_db(shard_df, collection_name=col_name, batch_size=1000)
        
        del shard_df
        gc.collect()
        
    print(f"\n[DONE] Shard(s) {to_process} build complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Sharded Vector DB")
    parser.add_argument("--shard", type=str, choices=["legacy", "mid", "recent", "all"], default="all", 
                        help="Choose which shard to build: legacy, mid, recent, or all")
    args = parser.parse_args()
    
    build_db(target_shard=args.shard)
