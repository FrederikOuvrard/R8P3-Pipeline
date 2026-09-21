from typing import List, Dict, Any, Optional
from pathlib import Path
import tantivy
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
        self.tantivy_index = None

        logger.info(f"Initialized R8P3 Hybrid Search Engine | Tantivy: {self.tantivy_index_path} | Qdrant Mode: {'Local Path (' + str(self.qdrant_local_path) + ')' if self.qdrant_local_path else 'Server'}")

    def initialize_indices(self) -> bool:
        try:
            self.tantivy_index_path.mkdir(parents=True, exist_ok=True)
            
            # Configuration du schéma Tantivy pour les documents juridiques
            schema_builder = tantivy.SchemaBuilder()
            schema_builder.add_text_field("document_name", stored=True)
            schema_builder.add_text_field("text", stored=True)
            schema = schema_builder.build()

            if any(self.tantivy_index_path.iterdir()):
                self.tantivy_index = tantivy.Index.open(str(self.tantivy_index_path))
                logger.info("Index Tantivy existant ouvert avec succès.")
            else:
                self.tantivy_index = tantivy.Index(schema, path=str(self.tantivy_index_path))
                logger.info("Nouvel index Tantivy créé.")

            # Initialisation Qdrant
            if self.qdrant_local_path:
                self.qdrant_local_path.parent.mkdir(parents=True, exist_ok=True)
                self.qdrant_client = QdrantClient(path=str(self.qdrant_local_path))
            else:
                self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

            collections = self.qdrant_client.get_collections().collections
            if not any(col.name == self.collection_name for col in collections):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
                )
                logger.info(f"Collection Qdrant '{self.collection_name}' créée.")

            return True
        except Exception as e:
            logger.exception(f"Erreur d'initialisation des index : {e}")
            return False

    def index_document(self, doc_name: str, text: str):
        """Indexe un document dans Tantivy."""
        if not self.tantivy_index:
            return
        writer = self.tantivy_index.writer()
        writer.add_document(tantivy.Document(document_name=[doc_name], text=[text]))
        writer.commit()

    def search_exact(self, query_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        logger.debug(f"Recherche exacte Tantivy pour : '{query_term}'")
        if not self.tantivy_index:
            return []
        try:
            searcher = self.tantivy_index.searcher()
            query = self.tantivy_index.parse_query(query_term, ["text"])
            hits = searcher.search(query, limit=limit)
            
            results = []
            for score, doc_address in hits:
                doc = searcher.doc(doc_address)
                results.append({
                    "document_name": doc.get("document_name", ["Inconnu"])[0],
                    "text": doc.get("text", [""])[0],
                    "score": float(score)
                })
            return results
        except Exception as e:
            logger.error(f"Erreur recherche Tantivy : {e}")
            return []

    def search_semantic(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        if not self.qdrant_client:
            return []
        try:
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return [dict(res) for res in results]
        except Exception as e:
            logger.error(f"Erreur recherche Qdrant : {e}")
            return []

    def hybrid_fusion_ranking(self, exact_results: List[Dict], semantic_results: List[Dict], alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Fusion RRF des résultats lexicaux et sémantiques."""
        combined = {}
        for res in exact_results:
            name = res.get("document_name")
            combined[name] = {"document_name": name, "text": res.get("text"), "score": res.get("score", 0.0) * alpha}
        return list(combined.values())
