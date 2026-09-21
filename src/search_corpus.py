# -*- coding: utf-8 -*-
"""
Module de Recherche Hybride CLI - R8P3-Pipeline
Version propre et unifiée combinant Tantivy (FTS) + Qdrant (Sémantique) avec RRF.
"""

import sys
from pathlib import Path
from loguru import logger
import tantivy
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

class LocalEmbedder:
    """Gère l'encodage sémantique local des requêtes."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> list:
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs[0]
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask)[0].tolist()

def hybrid_search(query: str, top_k: int = 5):
    tantivy_path = Path("indexes/tantivy")
    qdrant_path = Path("indexes/qdrant")

    if not tantivy_path.exists() or not qdrant_path.exists():
        logger.error("Les index Tantivy ou Qdrant sont introuvables. Veuillez lancer l'indexation.")
        return

    print(f"\nRecherche hybride en cours pour la requête : « {query} »...\n" + "-"*60)

    # 1. Recherche Sémantique (Qdrant)
    embedder = LocalEmbedder()
    query_vector = embedder.encode(query)
    
    qdrant_client = QdrantClient(path=str(qdrant_path))
    qdrant_response = qdrant_client.query_points(
        collection_name="historical_legal_corpus",
        query=query_vector,
        limit=top_k * 2
    )
    qdrant_results = qdrant_response.points

    # 2. Recherche Textuelle (Tantivy)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("filename", stored=True)
    schema_builder.add_unsigned_field("page_num", stored=True)
    schema_builder.add_text_field("body", stored=True)
    schema = schema_builder.build()

    index = tantivy.Index(schema, path=str(tantivy_path))
    searcher = index.searcher()
    tantivy_query = index.parse_query(query, ["body"])
    tantivy_hits = searcher.search(tantivy_query, limit=top_k * 2).hits

    # 3. Fusion des scores (RRF)
    scores = {}
    doc_metadata = {}

    for rank, hit in enumerate(qdrant_results):
        payload = hit.payload
        doc_id = (payload["filename"], payload["page_num"])
        doc_metadata[doc_id] = payload.get("body", "")
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (60 + rank + 1))

    for rank, (score, doc_address) in enumerate(tantivy_hits):
        doc = searcher.doc(doc_address)
        filename = doc.get_first("filename")
        page_num = doc.get_first("page_num")
        body = doc.get_first("body")
        doc_id = (filename, page_num)
        doc_metadata[doc_id] = body
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (60 + rank + 1))

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(f"=== TOP {top_k} RÉSULTATS HYBRIDES PERTINENTS ===")
    for idx, ((filename, page_num), rrf_score) in enumerate(sorted_results, 1):
        snippet = doc_metadata.get((filename, page_num), "")[:300].replace("\n", " ")
        print(f"\n[{idx}] Fichier : {filename} | Page : {page_num} (Score RRF : {rrf_score:.4f})")
        print(f"Extrait : {snippet}...")
        print("-" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recherche hybride dans le corpus juridique")
    parser.add_argument("query", type=str, help="Votre question ou termes de recherche juridique")
    parser.add_argument("-k", type=int, default=5, help="Nombre de résultats")
    args = parser.parse_args()
    
    hybrid_search(args.query, top_k=args.k)