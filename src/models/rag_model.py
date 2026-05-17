import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from src.core.config import settings

class RAGMarketMemory:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.client = chromadb.Client(Settings(persist_directory=settings.CHROMA_PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(settings.COLLECTION_NAME)

    def _create_text_chunk(self, row: pd.Series) -> str:
        """
        Converts market state into a descriptive text chunk.
        """
        ema_diff = row['ema10'] - row['ema50']
        return f"RSI {row['rsi']:.1f}, MACD {row['macd']:.4f}, ATR {row['atr']:.2f}, EMA diff {ema_diff:.2f}, returns {row['returns']:.5f}"

    def build_memory(self, df: pd.DataFrame):
        """
        Embeds historical data into ChromaDB.
        """
        texts = [self._create_text_chunk(row) for _, row in df.iterrows()]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Batch insert to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{"timestamp": str(idx), "target": int(row['target'])} for idx, row in df.iterrows()],
            ids=[f"idx_{i}" for i in range(len(df))]
        )

    def query_state(self, current_state: pd.Series, timestamp: str, top_k: int = 5) -> dict:
        """
        Queries the vector store for similar past states (strictly prior to current timestamp).
        """
        query_text = self._create_text_chunk(current_state)
        query_emb = self.model.encode([query_text])
        
        results = self.collection.query(
            query_embeddings=query_emb.tolist(),
            n_results=top_k + 1,
            where={"timestamp": {"$lt": timestamp}}
        )
        
        sim_targets = [m['target'] for m in results['metadatas'][0] if m['timestamp'] != timestamp][:top_k]
        
        if not sim_targets:
            sim_targets = [0]
            
        return {
            'sim_win_rate': np.mean(sim_targets),
            'sim_best_similarity': 1 - results['distances'][0][0] # Convert L2 to similarity
        }
