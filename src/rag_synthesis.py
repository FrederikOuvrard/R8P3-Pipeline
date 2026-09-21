# -*- coding: utf-8 -*-
"""
Module de Synthèse RAG Local - R8P3-Pipeline
Architecture propre avec Recherche Hybride (Tantivy + Qdrant + RRF) 
et Génération Doctrinale via Ollama en flux continu (Streaming).
"""

import sys
from pathlib import Path
import requests
import json
from loguru import logger
import tantivy
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

class LocalEmbedder:
    """Gère l'encodage sémantique local des requêtes via HuggingFace."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Chargement du modèle d'embedding : {model_name}")
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

class HybridRetriever:
    """Moteur de recherche hybride combinant Tantivy (FTS) et Qdrant (Sémantique) avec RRF."""
    def __init__(self, tantivy_dir: Path, qdrant_dir: Path):
        self.tantivy_path = tantivy_dir
        self.qdrant_path = qdrant_dir
        self.embedder = LocalEmbedder()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.tantivy_path.exists() or not self.qdrant_path.exists():
            logger.error("Les index Tantivy ou Qdrant sont introuvables. Veuillez lancer l'indexation.")
            return []

        # 1. Recherche Sémantique Vectorielle (Qdrant)
        query_vector = self.embedder.encode(query)
        qdrant_client = QdrantClient(path=str(self.qdrant_path))
        qdrant_response = qdrant_client.query_points(
            collection_name="historical_legal_corpus",
            query=query_vector,
            limit=top_k * 2
        )
        qdrant_results = qdrant_response.points

        # 2. Recherche Textuelle de Précision (Tantivy)
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("filename", stored=True)
        schema_builder.add_unsigned_field("page_num", stored=True)
        schema_builder.add_text_field("body", stored=True)
        schema = schema_builder.build()

        index = tantivy.Index(schema, path=str(self.tantivy_path))
        searcher = index.searcher()
        tantivy_query = index.parse_query(query, ["body"])
        tantivy_hits = searcher.search(tantivy_query, limit=top_k * 2).hits

        # 3. Fusion des scores par Reciprocal Rank Fusion (RRF)
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

        results = []
        for (filename, page_num), score in sorted_results:
            snippet = doc_metadata.get((filename, page_num), "")
            results.append({
                "filename": filename,
                "page_num": page_num,
                "score": score,
                "snippet": snippet
            })
        return results

def generate_legal_synthesis(query: str, model_name: str = "qwen-principal"):
    """Ordonne la recherche documentaire et la rédaction de la synthèse par Ollama en streaming."""
    print(f"\n[1/2] Analyse et recherche hybride pour : « {query} »...")
    retriever = HybridRetriever(Path("indexes/tantivy"), Path("indexes/qdrant"))
    contexts = retriever.search(query, top_k=5)

    if not contexts:
        print("Aucun contexte pertinent trouvé dans le corpus.")
        return

    context_block = "\n".join([
        f"--- SOURCE : {c['filename']} (Page {c['page_num']}) ---\n{c['snippet']}\n"
        for c in contexts
    ])

    prompt = f"""Tu es un assistant de recherche juridique hautement qualifié, spécialisé en droit civil et histoire du droit. 
Rédige une note de synthèse doctrinale rigoureuse, structurée et académique en réponse au sujet suivant : "{query}".

Règles impératives :
1. Appuie-toi EXCLUSIVEMENT sur les extraits historiques fournis ci-dessous. N'invente aucune information.
2. Cite explicitement les noms de fichiers et les numéros de page pour chaque affirmation ou doctrine mentionnée.
3. Adopte un ton formel, analytique et propre à la recherche universitaire en droit.

--- EXTRAITS DU CORPUS ---
{context_block}
--- FIN DES EXTRAITS ---

Rédige ta note de synthèse :"""

    print(f"[2/2] Génération de la note de synthèse via Ollama ({model_name}) en flux continu...\n")
    print("="*80)
    print(f"🏛️ NOTE DE SYNTHÈSE DOCTRINALE : {query}")
    print("="*80 + "\n")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": True},
            stream=True,
            timeout=300
        )
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    body = json.loads(line.decode('utf-8'))
                    chunk = body.get("response", "")
                    print(chunk, end="", flush=True)
            print("\n\n" + "="*80)
        else:
            logger.error(f"Erreur Ollama (Code {response.status_code})")
    except Exception as e:
        logger.error(f"Erreur de communication avec Ollama : {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Générateur de synthèse RAG pour corpus juridique")
    parser.add_argument("query", type=str, help="Sujet ou question juridique à synthétiser")
    parser.add_argument("-m", type=str, default="qwen-principal", help="Modèle Ollama")
    args = parser.parse_args()
    
    generate_legal_synthesis(args.query, model_name=args.m)