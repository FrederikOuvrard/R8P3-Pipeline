# -*- coding: utf-8 -*-
"""
Streamlit Research Workspace for R8P3-Pipeline
Clean local web interface for hybrid search and Markdown dossier generation.
"""

import streamlit as st
from pathlib import Path
import tantivy
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
import datetime

st.set_page_config(
    page_title="R8P3 - Atelier de Recherche Juridique",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_embedder():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def encode_query(query, tokenizer, model):
    inputs = tokenizer(query, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs[0]
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask)[0].tolist()

def run_hybrid_search(query, top_k=5):
    tantivy_path = Path("indexes/tantivy")
    qdrant_path = Path("indexes/qdrant")

    if not tantivy_path.exists() or not qdrant_path.exists():
        st.error("Les index Tantivy ou Qdrant sont introuvables. Veuillez lancer l'indexation.")
        return []

    tokenizer, model = load_embedder()
    query_vector = encode_query(query, tokenizer, model)

    # 1. Recherche Sémantique (Qdrant)
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
    
    results = []
    for (filename, page_num), rrf_score in sorted_results:
        snippet = doc_metadata.get((filename, page_num), "")
        results.append({
            "filename": filename,
            "page_num": page_num,
            "score": rrf_score,
            "snippet": snippet
        })
    return results

# --- Interface Streamlit ---
st.title("🏛️ R8P3 — Atelier de Recherche Juridique Historique")
st.markdown("Exploration souveraine du corpus de traités anciens (Fonds Hulot & Doctrine de Droit).")

query = st.text_input("🔍 Requête de recherche ou concept juridique (ex: *possession*, *quasi-contrats*):", "")
top_k = st.slider("Nombre de résultats à afficher", min_value=3, max_value=20, value=5)

if st.button("Lancer la recherche hybride", type="primary"):
    if query.strip():
        with st.spinner("Recherche conjointe dans Tantivy et Qdrant en cours..."):
            results = run_hybrid_search(query, top_k=top_k)
            
            if results:
                st.success(f"{len(results)} résultats pertinents trouvés pour : « {query} »")
                
                dossier_content = f"# Dossier Documentaire : {query}\n"
                dossier_content += f"*Généré le {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} via R8P3-Pipeline*\n\n---\n\n"

                for idx, res in enumerate(results, 1):
                    with st.expander(f"[{idx}] Fichier : `{res['filename']}` — Page : **{res['page_num']}** *(Score RRF : {res['score']:.4f})*"):
                        st.text_area(f"Extrait textuel ({res['filename']} - p.{res['page_num']})", res['snippet'], height=150, key=f"txt_{idx}")
                        
                    dossier_content += f"## [{idx}] Source : {res['filename']} (Page {res['page_num']})\n"
                    dossier_content += f"**Score RRF** : `{res['score']:.4f}`\n\n"
                    dossier_content += f"> {res['snippet'].replace('\n', ' ')}\n\n---\n\n"

                st.download_button(
                    label="📥 Télécharger le dossier documentaire (Markdown)",
                    data=dossier_content,
                    file_name=f"dossier-{query.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
            else:
                st.warning("Aucun résultat trouvé.")
    else:
        st.warning("Veuillez saisir une requête de recherche.")