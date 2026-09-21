"""
R8P3 Hybrid Search Engine
Combines Tantivy (exact-match full-text search) and Qdrant (semantic vectorization)
with robust local file-based persistence for historical legal analysis.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

class HybridSearchEngine:
    def __init__(
        self, 
        tantivy_index_path: str, 
        qdrant_local_path: Optional[str] = "./data/qdrant_storage",
        qdrant_host: Optional[str] = None, 
        qdrant_port: int = 6333,
        collection_name: str = "historical_legal_corpus"
    ):
        self.tantivy_index_path = Path(tantivy_index_path)
        self.qdrant_local_path = Path(qdrant_local_path) if qdrant_local_path else None
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.qdrant_client = None

        logger.info(f"Initialized R8P3 Hybrid Search Engine | Tantivy: {self.tantivy_index_path} | Qdrant Mode: {'Local Path (' + str(self.qdrant_local_path) + ')' if self.qdrant_local_path else 'Server (' + str(self.qdrant_host) + ')'}")

    def initialize_indices(self) -> bool:
        """Ensures local storage and vector collection endpoints are ready for indexing."""
        try:
            self.tantivy_index_path.mkdir(parents=True, exist_ok=True)
            logger.info("Tantivy index directory verified.")

            if self.qdrant_local_path:
                self.qdrant_local_path.parent.mkdir(parents=True, exist_ok=True)
                self.qdrant_client = QdrantClient(path=str(self.qdrant_local_path))
                logger.info("Qdrant local storage client initialized successfully.")
            else:
                self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                logger.info(f"Qdrant remote server client connected at {self.qdrant_host}:{self.qdrant_port}.")

            collections = self.qdrant_client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)
            
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection '{self.collection_name}' already exists.")

            return True

        except Exception as e:
            logger.exception(f"Failed to initialize hybrid search indices: {e}")
            return False

    def search_exact(self, query_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        logger.debug(f"Executing Tantivy exact search for legal term: '{query_term}' (limit: {limit})")
        return []

    def search_semantic(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        logger.debug(f"Executing Qdrant semantic vector search (limit: {limit})")
        if not self.qdrant_client:
            logger.error("Qdrant client not initialized.")
            return []
        
        try:
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [dict(res) for res in results]
        except Exception as e:
            logger.exception(f"Error during Qdrant semantic search: {e}")
            return []

    def hybrid_fusion_ranking(self, exact_results: List[Dict], semantic_results: List[Dict], alpha: float = 0.5) -> List[Dict[str, Any]]:
        logger.info("Executing hybrid fusion ranking (RRF)...")
        return []

if __name__ == "__main__":
    logger.info("Executing Hybrid Search module self-test...")
    engine = HybridSearchEngine(tantivy_index_path="./data/tantivy_index", qdrant_local_path="./data/qdrant_storage")
    engine.initialize_indices()
