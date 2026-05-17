import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
import time
import os
import psutil
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class MarketRAG:
    def __init__(self, db_dir: str = "./data/chromadb"):
        # GPU Optimization (AMD DirectML or NVIDIA CUDA)
        self.device = "cpu"
        try:
            import torch_directml
            if torch_directml.is_available():
                self.device = torch_directml.device()
                print(f"AMD GPU Detected: Using {torch_directml.device_name(0)} via DirectML")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("NVIDIA GPU Detected: Using CUDA")
        except Exception as e:
            print(f"GPU Mode Fallback to CPU: {e}")
            pass
            
        print(f"MarketRAG using device: {self.device}")
        
        # THERMAL SAFETY: Limit CPU threads to prevent 100% sustained load
        if self.device == "cpu":
            num_cores = os.cpu_count() or 4
            safe_threads = max(1, num_cores // 2) # Use half the cores
            torch.set_num_threads(safe_threads)
            print(f"Thermal Safety: Limiting torch to {safe_threads} threads.")

        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # RESOURCE CAPS (R1 mitigation)
        self.max_ram_gb = 14.0  # Fail/Warn if > 14GB
        self.max_gpu_temp = 85  # Thermal safety (R7)
        if HAS_NVML:
            try: nvmlInit()
            except: pass
        
        # Pass a dummy embedding function to avoid the default ONNX runtime initialization issue
        class DummyEF:
            def __call__(self, input): return [[0.0] * 384] * len(input)
        self.dummy_ef = DummyEF()
            
        # Sharded Collections (Lazy Loaded)
        self.collection_names = ["gold_legacy", "gold_mid", "gold_recent"]
        self._collections = {}
        self.master_csv = "./data/xauusd_features.csv" # For R4 fallback
        print(f"MarketRAG initialized. Collections will be lazy-loaded.")

    def get_collection(self, name: str):
        if name not in self.collection_names:
            raise ValueError(f"Invalid collection name: {name}")
        
        if name not in self._collections:
            print(f"Loading/Creating collection: {name}...")
            # HNSW M=16 optimization for memory efficiency (Audit R1)
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.dummy_ef,
                metadata={
                    "hnsw:space": "cosine", 
                    "hnsw:M": 16, 
                    "hnsw:construction_ef": 100
                }
            )
        return self._collections[name]

    def check_resources(self):
        """
        Monitors RAM and GPU temperature for thermal and memory safety (R1, R7).
        """
        # RAM Check
        ram_used = psutil.virtual_memory().used / (1024**3)
        if ram_used > self.max_ram_gb:
            print(f"CRITICAL: Memory limit exceeded ({ram_used:.1f}GB > {self.max_ram_gb}GB)")
            import gc
            gc.collect()
            if ram_used > self.max_ram_gb + 1: # Hard cap
                 raise MemoryError(f"System RAM exceeded safety limit: {ram_used:.1f}GB")

        # GPU Temp Check
        if HAS_NVML and self.device == "cuda":
            try:
                handle = nvmlDeviceGetHandleByIndex(0)
                temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                if temp > self.max_gpu_temp:
                    print(f"THERMAL ALERT: GPU Temp {temp}°C > {self.max_gpu_temp}°C. Throttling...")
                    return True # Throttling needed
            except:
                pass
        return False

    # =================================================================
    # PHASE 4.3: Enhanced row_to_text with session, trend, volatility
    # =================================================================
    def row_to_text(self, row: pd.Series, macro_snippet: str = "no major news") -> str:
        """
        Converts indicator + price state into a RICH textual description.
        Includes: core indicators, momentum, trend regime, session, candle pattern.
        """
        # Core indicators
        ema_rel = "above" if row.get('ema10', 0) > row.get('ema50', 0) else "below"
        rsi = row.get('rsi', 50)
        
        # RSI zone classification
        if rsi > 70:
            rsi_zone = "overbought"
        elif rsi > 60:
            rsi_zone = "bullish"
        elif rsi > 40:
            rsi_zone = "neutral"
        elif rsi > 30:
            rsi_zone = "bearish"
        else:
            rsi_zone = "oversold"
        
        # Trend alignment (from feature engineering)
        trend_val = row.get('trend_alignment', 0)
        if trend_val > 0.5:
            trend = "strong uptrend"
        elif trend_val > 0:
            trend = "weak uptrend"
        elif trend_val > -0.5:
            trend = "weak downtrend"
        else:
            trend = "strong downtrend"
        
        # Volatility regime (from ATR percentile)
        atr_pct = row.get('atr_percentile', 0.5)
        if atr_pct > 0.8:
            vol_regime = "high volatility"
        elif atr_pct > 0.5:
            vol_regime = "normal volatility"
        else:
            vol_regime = "low volatility"
        
        # Bollinger squeeze
        squeeze = "squeeze active" if row.get('bb_squeeze', 0) > 0 else "no squeeze"
        
        # Candle pattern
        body = row.get('body_ratio', 0.5)
        upper_wick = row.get('upper_wick_ratio', 0)
        lower_wick = row.get('lower_wick_ratio', 0)
        
        if body < 0.2:
            candle = "doji"
        elif upper_wick > 0.5:
            candle = "shooting star"
        elif lower_wick > 0.5:
            candle = "hammer"
        elif body > 0.7:
            candle = "strong body"
        else:
            candle = "normal candle"
        
        # Momentum
        mom5 = row.get('momentum_5', 0) * 100
        
        # Session detection from timestamp (if available)
        session = "unknown"
        try:
            ts = pd.to_datetime(row.name) if hasattr(row, 'name') else None
            if ts is not None:
                hour = ts.hour
                if 0 <= hour < 8:
                    session = "asian"
                elif 8 <= hour < 14:
                    session = "london"
                elif 14 <= hour < 21:
                    session = "newyork"
                else:
                    session = "late"
        except:
            pass
        
        return (
            f"RSI {rsi:.1f} ({rsi_zone}), "
            f"MACD hist {row.get('macd_hist', 0):.3f}, "
            f"ATR {row.get('atr', 0):.2f}, "
            f"EMA10 {ema_rel} EMA50, "
            f"return {row.get('returns', 0)*100:.2f}%, "
            f"momentum {mom5:.2f}%, "
            f"trend {trend}, "
            f"{vol_regime}, {squeeze}, "
            f"candle {candle}, "
            f"session {session}, "
            f"bb_pos {row.get('bb_position', 0.5):.2f}, "
            f"{macro_snippet}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def add_to_db(self, df: pd.DataFrame, collection_name: str, batch_size: int = 1000):
        """
        Adds historical data to a specific ChromaDB collection in batches.
        Now stores volatility regime metadata for filtered retrieval.
        """
        target_collection = self.get_collection(collection_name)

        total = len(df)
        for batch_start in range(0, total, batch_size):
            batch_df = df.iloc[batch_start:batch_start+batch_size]
            texts = [self.row_to_text(row) for _, row in batch_df.iterrows()]
            embeddings = self.embed(texts)
            
            timestamps = [pd.to_datetime(idx).timestamp() for idx in batch_df.index]
            
            metadatas = []
            for j, (idx, row) in enumerate(batch_df.iterrows()):
                # Classify volatility regime for filtered RAG (Phase 4.2)
                atr_pct = row.get('atr_percentile', 0.5)
                if atr_pct > 0.7:
                    regime = "high_vol"
                elif atr_pct > 0.3:
                    regime = "normal_vol"
                else:
                    regime = "low_vol"
                
                metadatas.append({
                    "timestamp": float(timestamps[j]),
                    "timestamp_str": str(idx),
                    "actual_next_move": int(row['next_5m_direction']),
                    "actual_return": float(row['next_return']),
                    "regime": regime
                })
                
            ids = [f"id_{ts}" for ts in timestamps]
            
            target_collection.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            # THERMAL & RESOURCE MONITORING
            throttle = self.check_resources()
            
            print(f"Added batch {batch_start} to {batch_start+len(batch_df)} / {total} to {collection_name}.")
            
            # Explicitly clear batch data to free memory
            del texts, embeddings, metadatas, ids, timestamps
            import gc
            gc.collect()
            
            # THERMAL SAFETY: Sleep for 2 seconds (or 10 if throttling)
            sleep_time = 10 if throttle else 2
            print(f"Thermal Safety: Cooling down for {sleep_time} seconds...")
            time.sleep(sleep_time)

    # =================================================================
    # PHASE 4.1 + 4.2: Recency-weighted, regime-filtered retrieval
    # =================================================================
    def retrieve_similar(self, current_row: dict, current_timestamp_str: str, 
                         macro_snippet: str = "no major news", top_k: int = 5,
                         regime_filter: str = None, recency_weight: float = 0.15) -> dict:
        """
        Retrieves similar past patterns from ALL sharded collections.
        
        Enhancements (Phase 4):
        - recency_weight: Applies exponential decay so recent matches rank higher.
          0.0 = pure similarity, 1.0 = heavily favor recent data.
        - regime_filter: If set ("high_vol", "normal_vol", "low_vol"), only retrieves
          patterns from the same volatility regime.
        """
        query_text = self.row_to_text(pd.Series(current_row), macro_snippet)
        query_emb = self.embed([query_text])
        
        # Robust Timezone Fix: Handle any input (IST, Offset, etc.)
        ts_obj = pd.to_datetime(current_timestamp_str)
        if ts_obj.tzinfo is not None:
            ts_obj = ts_obj.tz_convert('UTC').tz_localize(None)
        current_ts = ts_obj.timestamp()
        
        all_metas = []
        all_distances = []
        all_docs = []
        
        # Build where filter
        where_conditions = {"timestamp": {"$lt": current_ts}}
        
        # Phase 4.2: Regime-filtered retrieval
        if regime_filter:
            where_conditions = {
                "$and": [
                    {"timestamp": {"$lt": current_ts}},
                    {"regime": regime_filter}
                ]
            }
        
        # Query each shard
        for name in self.collection_names:
            try:
                col = self.get_collection(name)
                results = col.query(
                    query_embeddings=query_emb,
                    n_results=top_k + 5,  # Fetch small extra for recency re-ranking
                    where=where_conditions
                )
                
                if results['metadatas'][0]:
                    all_metas.extend(results['metadatas'][0])
                    all_distances.extend(results['distances'][0])
                    all_docs.extend(results['documents'][0])
            except Exception as e:
                # Graceful fallback if regime filter fails (e.g., old data without regime)
                try:
                    results = col.query(
                        query_embeddings=query_emb,
                        n_results=top_k,
                        where={"timestamp": {"$lt": current_ts}}
                    )
                    if results['metadatas'][0]:
                        all_metas.extend(results['metadatas'][0])
                        all_distances.extend(results['distances'][0])
                        all_docs.extend(results['documents'][0])
                except:
                    pass
        
        if not all_metas:
            return {
                "sim_win_rate": 0.5,
                "sim_avg_return": 0.0,
                "sim_max_similarity": 0.0,
                "similar_patterns": [],
                "regime_used": regime_filter or "none"
            }
        
        # =============================================================
        # Phase 4.1: RECENCY-WEIGHTED re-ranking
        # Combine cosine distance with time decay for smarter ranking
        # =============================================================
        combined = []
        for dist, meta, doc in zip(all_distances, all_metas, all_docs):
            # Cosine similarity (higher = better)
            cosine_sim = 1.0 / (1.0 + dist)
            
            # Recency score: exponential decay based on time gap
            time_gap_days = (current_ts - meta['timestamp']) / 86400.0  # Convert to days
            recency_score = np.exp(-time_gap_days / 365.0)  # 1-year half-life
            
            # Blended score (higher = better match)
            blended = (1.0 - recency_weight) * cosine_sim + recency_weight * recency_score
            
            combined.append((blended, dist, meta, doc))
        
        # Sort by blended score DESCENDING (highest = best)
        combined.sort(key=lambda x: x[0], reverse=True)
        combined = combined[:top_k]
        
        final_blended, final_distances, final_metas, final_docs = zip(*combined)
        
        win_rate = np.mean([m['actual_next_move'] for m in final_metas])
        avg_ret = np.mean([m['actual_return'] for m in final_metas])
        max_sim = max(final_blended)
        
        patterns = []
        for i in range(len(final_metas)):
            patterns.append({
                "document": final_docs[i],
                "timestamp": final_metas[i].get('timestamp_str', str(final_metas[i]['timestamp'])),
                "move": final_metas[i]['actual_next_move'],
                "similarity": float(final_blended[i]),
                "regime": final_metas[i].get('regime', 'unknown'),
                "raw_cosine": 1.0 / (1.0 + final_distances[i])
            })
            
        return {
            "sim_win_rate": float(win_rate),
            "sim_avg_return": float(avg_ret),
            "sim_max_similarity": float(max_sim),
            "similar_patterns": patterns,
            "regime_used": regime_filter or "auto"
        }

    # =============================================================
    # PHASE 4.4: CORRUPTION RECOVERY & FALLBACK (R4)
    # =============================================================
    def backup_db(self, backup_dir: str = "./data/chromadb_backups"):
        """
        Creates a timestamped checkpoint of the ChromaDB directory.
        """
        import shutil
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        target = os.path.join(backup_dir, f"chroma_backup_{ts}")
        try:
            shutil.copytree(self.client._settings.persist_directory, target)
            print(f"Backup successful: {target}")
        except Exception as e:
            print(f"Backup failed: {e}")

    def search_csv_fallback(self, current_row: dict, top_k: int = 5) -> dict:
        """
        Statistical Baseline (R5): If ChromaDB is down/corrupt, search the master CSV
        using a simple Euclidean distance on key technical features.
        """
        if not os.path.exists(self.master_csv):
            return {"error": "Master CSV not found for fallback."}
            
        print("RAG FALLBACK: Searching master CSV for similar patterns...")
        try:
            # Load a sample to keep memory low (or use the full one if sharded)
            # For brevity, we assume the CSV is accessible. In production, we might use an index.
            df = pd.read_csv(self.master_csv, index_col=0, nrows=50000) # Recent 50k
            
            # Key features for distance
            match_cols = ['rsi', 'macd_hist', 'atr', 'returns']
            X_past = df[match_cols].values
            X_curr = np.array([[current_row.get(c, 0) for c in match_cols]])
            
            # Simple Euclidean distance
            distances = np.linalg.norm(X_past - X_curr, axis=1)
            closest_idx = np.argsort(distances)[:top_k]
            
            results = df.iloc[closest_idx]
            win_rate = (results['next_5m_direction'] > 0).mean()
            avg_ret = results['next_return'].mean()
            
            return {
                "sim_win_rate": float(win_rate),
                "sim_avg_return": float(avg_ret),
                "sim_max_similarity": 0.5, # Placeholder for fallback
                "is_fallback": True
            }
        except Exception as e:
            return {"error": f"CSV Fallback failed: {e}"}
