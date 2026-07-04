import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import time
import os
import shutil
import psutil
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

class MarketRAG:
    def __init__(self, index_dir: str = "./data/faiss"):
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
        
        # FAISS Index Paths
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "market_memory.index")
        self.metadata_path = os.path.join(index_dir, "metadata.parquet")
        
        # RESOURCE CAPS (R1 mitigation)
        self.max_ram_gb = 15.0  # Increased to avoid false positives if system baseline RAM is high
        self.max_gpu_temp = 85  # Thermal safety (R7)
        if HAS_NVML:
            try: nvmlInit()
            except: pass
        
        # Lazy-loaded FAISS index and metadata
        self._index = None
        self._metadata = None
        self.master_csv = "./data/xauusd_features.csv" # For R4 fallback
        
        print(f"MarketRAG initialized (FAISS backend). Index will be lazy-loaded from {index_dir}.")

    def _load_index(self):
        """Lazily loads the FAISS index and metadata into memory."""
        if self._index is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(
                    f"FAISS index not found at {self.index_path}. "
                    f"Run `python -m rag.build_vector_db` to build it."
                )
            print(f"Loading FAISS index from {self.index_path}...")
            self._index = faiss.read_index(self.index_path)
            # Set search-time nprobe for IVF indexes (controls speed vs accuracy tradeoff)
            if hasattr(self._index, 'nprobe'):
                self._index.nprobe = 32  # Search 32 of 1024 clusters (good balance)
            print(f"  FAISS index loaded: {self._index.ntotal} vectors, dimension={self._index.d}")
        
        if self._metadata is None:
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(
                    f"Metadata file not found at {self.metadata_path}. "
                    f"Run `python -m rag.build_vector_db` to build it."
                )
            print(f"Loading metadata from {self.metadata_path}...")
            self._metadata = pd.read_parquet(self.metadata_path)
            print(f"  Metadata loaded: {len(self._metadata)} rows")
        
        return self._index, self._metadata

    @property
    def index_loaded(self) -> bool:
        """Check whether the FAISS index is loaded."""
        return self._index is not None

    @property
    def total_vectors(self) -> int:
        """Returns total vectors in the loaded index, or 0 if not loaded."""
        if self._index is not None:
            return self._index.ntotal
        return 0

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

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encodes texts to embeddings. Returns np.float32 array."""
        return self.model.encode(texts, convert_to_numpy=True).astype(np.float32)

    # =================================================================
    # INDEX BUILDING (replaces add_to_db)
    # =================================================================
    def build_index(self, df: pd.DataFrame, batch_size: int = 5000):
        """
        Builds a FAISS IVF-PQ index from the full DataFrame.
        Stores all vectors in a single compressed index file + metadata Parquet.
        
        This replaces the old add_to_db() method that wrote to ChromaDB collections.
        """
        os.makedirs(self.index_dir, exist_ok=True)
        total = len(df)
        print(f"\n=== Building FAISS Index for {total} rows ===")
        
        # --- Phase 1: Generate all embeddings in batches ---
        print("Phase 1: Generating embeddings...")
        all_embeddings = []
        all_texts = []
        
        for batch_start in range(0, total, batch_size):
            batch_df = df.iloc[batch_start:batch_start + batch_size]
            texts = [self.row_to_text(row) for _, row in batch_df.iterrows()]
            embeddings = self.embed(texts)
            
            all_embeddings.append(embeddings)
            all_texts.extend(texts)
            
            # Thermal & resource monitoring
            throttle = self.check_resources()
            print(f"  Embedded batch {batch_start} to {batch_start + len(batch_df)} / {total}")
            
            # Thermal safety cooldown
            sleep_time = 5 if throttle else 1
            time.sleep(sleep_time)
        
        embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
        dimension = embeddings_matrix.shape[1]  # 384 for all-MiniLM-L6-v2
        print(f"  Total embeddings: {embeddings_matrix.shape}")
        
        # --- Phase 2: Build metadata DataFrame ---
        print("Phase 2: Building metadata...")
        timestamps = []
        timestamp_strs = []
        actual_moves = []
        actual_returns = []
        regimes = []
        
        for idx, row in df.iterrows():
            ts_obj = pd.to_datetime(idx)
            timestamps.append(float(ts_obj.timestamp()))
            timestamp_strs.append(str(idx))
            actual_moves.append(int(row['next_5m_direction']))
            actual_returns.append(float(row['next_return']))
            
            # Classify volatility regime
            atr_pct = row.get('atr_percentile', 0.5)
            if atr_pct > 0.7:
                regimes.append("high_vol")
            elif atr_pct > 0.3:
                regimes.append("normal_vol")
            else:
                regimes.append("low_vol")
        
        metadata_df = pd.DataFrame({
            "timestamp": timestamps,
            "timestamp_str": timestamp_strs,
            "actual_next_move": actual_moves,
            "actual_return": actual_returns,
            "regime": regimes,
            "document": all_texts
        })
        
        # --- Phase 3: Train and build FAISS index ---
        print("Phase 3: Building FAISS index...")
        
        n_vectors = embeddings_matrix.shape[0]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        if n_vectors < 10000:
            print(f"  Small dataset ({n_vectors} vectors), using IndexFlatIP instead of IVF-PQ")
            index = faiss.IndexFlatIP(dimension)
        else:
            # IVF-PQ parameters tuned for 1.7M vectors, 384 dimensions
            # Number of IVF clusters: sqrt(N) is a good heuristic, capped reasonably
            n_clusters = min(int(np.sqrt(n_vectors)), 2048)
            n_clusters = max(n_clusters, 64)  # Minimum 64 clusters
            
            # PQ sub-quantizers: dimension must be divisible by this
            # 384 / 48 = 8 bytes per sub-quantizer
            pq_m = 48  # Number of sub-quantizers
            pq_bits = 8  # Bits per sub-quantizer code
            
            print(f"  IVF clusters: {n_clusters}, PQ sub-quantizers: {pq_m}, bits: {pq_bits}")
            
            # Build the index
            quantizer = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
            index = faiss.IndexIVFPQ(quantizer, dimension, n_clusters, pq_m, pq_bits)
            
            # Train on a representative sample (up to 100k vectors)
            train_size = min(n_vectors, 100000)
            train_sample = embeddings_matrix[
                np.random.choice(n_vectors, train_size, replace=False)
            ]
            
            print(f"  Training on {train_size} sample vectors...")
            index.train(train_sample)
            
        # Add all vectors
        print(f"  Adding {n_vectors} vectors to index...")
        index.add(embeddings_matrix)
        
        print(f"  Index built: {index.ntotal} vectors indexed")
        
        # --- Phase 4: Save ---
        print("Phase 4: Saving index and metadata...")
        faiss.write_index(index, self.index_path)
        metadata_df.to_parquet(self.metadata_path, index=False)
        
        index_size_mb = os.path.getsize(self.index_path) / (1024 * 1024)
        meta_size_mb = os.path.getsize(self.metadata_path) / (1024 * 1024)
        
        print(f"\n=== FAISS Index Build Complete ===")
        print(f"  Index file: {self.index_path} ({index_size_mb:.1f} MB)")
        print(f"  Metadata:   {self.metadata_path} ({meta_size_mb:.1f} MB)")
        print(f"  Total:      {index_size_mb + meta_size_mb:.1f} MB (vs ~9,700 MB with ChromaDB)")
        print(f"  Vectors:    {index.ntotal}")
        
        # Clear cached state to force reload on next query
        self._index = None
        self._metadata = None

    # =================================================================
    # PHASE 4.1 + 4.2: Recency-weighted, regime-filtered retrieval
    # =================================================================
    def retrieve_similar(self, current_row: dict, current_timestamp_str: str, 
                         macro_snippet: str = "no major news", top_k: int = 5,
                         regime_filter: str = None, recency_weight: float = 0.15) -> dict:
        """
        Retrieves similar past patterns using FAISS vector search.
        
        Enhancements (Phase 4):
        - recency_weight: Applies exponential decay so recent matches rank higher.
          0.0 = pure similarity, 1.0 = heavily favor recent data.
        - regime_filter: If set ("high_vol", "normal_vol", "low_vol"), only retrieves
          patterns from the same volatility regime.
          
        Returns the same dict format as the original ChromaDB implementation for
        full backward compatibility.
        """
        index, metadata = self._load_index()
        
        query_text = self.row_to_text(pd.Series(current_row), macro_snippet)
        query_emb = self.embed([query_text])
        
        # Normalize query for cosine similarity (index was built with normalized vectors)
        faiss.normalize_L2(query_emb)
        
        # Robust Timezone Fix: Handle any input (IST, Offset, etc.)
        ts_obj = pd.to_datetime(current_timestamp_str)
        if ts_obj.tzinfo is not None:
            ts_obj = ts_obj.tz_convert('UTC').tz_localize(None)
        current_ts = ts_obj.timestamp()
        
        # Fetch extra candidates for post-filtering (FAISS doesn't support WHERE clauses)
        # We fetch 10x the needed amount to ensure enough remain after timestamp/regime filtering
        fetch_k = min(top_k * 10, index.ntotal)
        
        distances, indices = index.search(query_emb, fetch_k)
        
        # Flatten results (search returns 2D arrays)
        distances = distances[0]
        indices = indices[0]
        
        # Filter out invalid indices (-1 means no result)
        valid_mask = indices >= 0
        distances = distances[valid_mask]
        indices = indices[valid_mask]
        
        if len(indices) == 0:
            return {
                "sim_win_rate": 0.5,
                "sim_avg_return": 0.0,
                "sim_max_similarity": 0.0,
                "similar_patterns": [],
                "regime_used": regime_filter or "none"
            }
        
        # Retrieve metadata for matched indices
        matched_meta = metadata.iloc[indices]
        
        # POST-FILTER 1: Strict timestamp boundary (prevent data leakage / look-ahead)
        time_mask = matched_meta['timestamp'].values < current_ts
        
        # POST-FILTER 2: Regime filter (Phase 4.2)
        if regime_filter:
            regime_mask = matched_meta['regime'].values == regime_filter
            combined_mask = time_mask & regime_mask
        else:
            combined_mask = time_mask
        
        # Apply filters
        filtered_distances = distances[combined_mask]
        filtered_meta = matched_meta[combined_mask]
        
        if len(filtered_meta) == 0:
            # Fallback: try without regime filter
            if regime_filter:
                filtered_distances = distances[time_mask]
                filtered_meta = matched_meta[time_mask]
            
            if len(filtered_meta) == 0:
                return {
                    "sim_win_rate": 0.5,
                    "sim_avg_return": 0.0,
                    "sim_max_similarity": 0.0,
                    "similar_patterns": [],
                    "regime_used": regime_filter or "none"
                }
        
        # =============================================================
        # Phase 4.1: RECENCY-WEIGHTED re-ranking
        # Combine cosine similarity with time decay for smarter ranking
        # =============================================================
        combined = []
        for i in range(len(filtered_meta)):
            meta_row = filtered_meta.iloc[i]
            dist = filtered_distances[i]
            
            # For IVF-PQ with inner product, higher distance = more similar
            cosine_sim = max(0.0, float(dist))
            
            # Recency score: exponential decay based on time gap
            time_gap_days = (current_ts - meta_row['timestamp']) / 86400.0
            recency_score = np.exp(-time_gap_days / 365.0)  # 1-year half-life
            
            # Blended score (higher = better match)
            blended = (1.0 - recency_weight) * cosine_sim + recency_weight * recency_score
            
            combined.append((blended, cosine_sim, meta_row))
        
        # Sort by blended score DESCENDING (highest = best)
        combined.sort(key=lambda x: x[0], reverse=True)
        combined = combined[:top_k]
        
        # Extract final results
        win_rate = np.mean([c[2]['actual_next_move'] for c in combined])
        avg_ret = np.mean([c[2]['actual_return'] for c in combined])
        max_sim = max(c[0] for c in combined)
        
        patterns = []
        for blended, raw_cosine, meta_row in combined:
            patterns.append({
                "document": meta_row.get('document', ''),
                "timestamp": meta_row.get('timestamp_str', str(meta_row['timestamp'])),
                "move": int(meta_row['actual_next_move']),
                "similarity": float(blended),
                "regime": meta_row.get('regime', 'unknown'),
                "raw_cosine": float(raw_cosine)
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
    def backup_db(self, backup_dir: str = "./data/faiss_backups"):
        """
        Creates a timestamped backup of the FAISS index and metadata files.
        Much simpler than ChromaDB — just 2 files to copy.
        """
        from datetime import datetime
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        target = os.path.join(backup_dir, f"faiss_backup_{ts}")
        os.makedirs(target, exist_ok=True)
        
        try:
            if os.path.exists(self.index_path):
                shutil.copy2(self.index_path, os.path.join(target, "market_memory.index"))
            if os.path.exists(self.metadata_path):
                shutil.copy2(self.metadata_path, os.path.join(target, "metadata.parquet"))
            print(f"Backup successful: {target}")
        except Exception as e:
            print(f"Backup failed: {e}")

    def search_csv_fallback(self, current_row: dict, top_k: int = 5) -> dict:
        """
        Statistical Baseline (R5): If FAISS index is missing/corrupt, search the master CSV
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
