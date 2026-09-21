"""
Hybrid Indexing Script for R8P3-Pipeline with Qdrant Embeddings
Indexes extracted historical legal texts into Tantivy (FTS) and Qdrant (Vector Search).
"""

import os
from pathlib import Path
from loguru import logger
import tantivy
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pymupdf

class LocalEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Chargement du modèle d'embedding : {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, text: str) -> list:
        # Tronquer et tokeniser pour le modèle (max 512 tokens)
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean Pooling pour obtenir un vecteur de document/page robuste
        token_embeddings = outputs[0]
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings[0].tolist()

def setup_tantivy_index(index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("filename", stored=True)
    schema_builder.add_unsigned_field("page_num", stored=True)
    schema_builder.add_text_field("body", stored=True)
    schema = schema_builder.build()
    
    try:
        index = tantivy.Index(schema, path=str(index_dir))
    except Exception:
        index = tantivy.Index(schema)
    return index

def setup_qdrant_client(db_path: Path):
    db_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(db_path))
    collection_name = "historical_legal_corpus"
    
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
        logger.info(f"Collection Qdrant '{collection_name}' créée (dimension 384).")
    return client, collection_name

def run_hybrid_indexing():
    corpus_dir = Path(r"D:\TheseIA\work\these-cowork\reports\fable-lots\codex-v42-condensation-corpus-20260912-r1\preuve-lot31-709-736\source-pdf")
    tantivy_path = Path("indexes/tantivy")
    qdrant_path = Path("indexes/qdrant")

    logger.info("Initialisation de l'indexation hybride (Tantivy + Qdrant + Embeddings)...")
    tantivy_index = setup_tantivy_index(tantivy_path)
    tantivy_writer = tantivy_index.writer()

    qdrant_client, qdrant_collection = setup_qdrant_client(qdrant_path)
    embedder = LocalEmbedder()

    pdf_files = sorted(list(corpus_dir.glob("*.pdf")))
    logger.info(f"Démarrage du traitement pour {len(pdf_files)} documents...")

    total_points = 0
    for pdf_path in pdf_files:
        logger.info(f"\n[Indexation] Traitement de : {pdf_path.name}")
        try:
            doc = pymupdf.open(str(pdf_path))
            qdrant_batch = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text()
                if not text.strip():
                    continue

                page_num = page_idx + 1

                # 1. Ajout Tantivy (Full-Text Search)
                tantivy_writer.add_document(tantivy.Document(
                    filename=[pdf_path.name],
                    page_num=[page_num],
                    body=[text]
                ))

                # 2. Génération de l'embedding pour Qdrant (Recherche sémantique)
                vector = embedder.encode(text)
                point_id = total_points + 1

                qdrant_batch.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "filename": pdf_path.name,
                            "page_num": page_num,
                            "body": text[:1000] # Stockage d'un extrait textuel de référence
                        }
                    )
                )
                total_points += 1

                # Insertion par batch de 32 pages pour optimiser la mémoire
                if len(qdrant_batch) >= 32:
                    qdrant_client.upsert(collection_name=qdrant_collection, points=qdrant_batch)
                    qdrant_batch = []

            # Insérer le reste du batch si besoin
            if qdrant_batch:
                qdrant_client.upsert(collection_name=qdrant_collection, points=qdrant_batch)

            tantivy_writer.commit()
            logger.info(f"[Succès] Document {pdf_path.name} indexé (Tantivy & Qdrant - {len(doc)} pages).")

        except Exception as e:
            logger.error(f"[Erreur] Échec sur {pdf_path.name}: {e}")

    logger.info(f"\n=== FIN DE L'INDEXATION HYBRIDE COMPLÈTE ===")
    logger.info(f"Total des segments vectorisés et indexés : {total_points}")

if __name__ == "__main__":
    run_hybrid_indexing()
