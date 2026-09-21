# -*- coding: utf-8 -*-
"""
Gestionnaire de projet R8P3-Pipeline
Vérifie l'intégrité de l'environnement, des index et des services locaux.
"""

from pathlib import Path
import requests
from loguru import logger

class ProjectManager:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.tantivy_path = self.root_dir / "indexes" / "tantivy"
        self.qdrant_path = self.root_dir / "indexes" / "qdrant"

    def check_indexes(self) -> bool:
        logger.info("Vérification des index locaux...")
        tantivy_exists = self.tantivy_path.exists()
        qdrant_exists = self.qdrant_path.exists()

        if tantivy_exists:
            logger.success(f"Index Tantivy trouvé : {self.tantivy_path}")
        else:
            logger.warning("Index Tantivy introuvable. Lancez `index_corpus.py`.")

        if qdrant_exists:
            logger.success(f"Index Qdrant trouvé : {self.qdrant_path}")
        else:
            logger.warning("Index Qdrant introuvable. Lancez `index_corpus.py`.")

        return tantivy_exists and qdrant_exists

    def check_ollama(self, model_name: str = "qwen-principal") -> bool:
        logger.info(f"Vérification du service Ollama (Modèle : {model_name})...")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if any(model_name in m for m in models):
                    logger.success(f"Ollama est actif et le modèle '{model_name}' est disponible.")
                    return True
                else:
                    logger.warning(f"Ollama répond, mais le modèle '{model_name}' est introuvable parmi : {models}")
                    return False
            else:
                logger.error(f"Ollama a répondu avec le code d'erreur {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Impossible de joindre Ollama sur http://localhost:11434 : {e}")
            return False

    def run_diagnostics(self):
        print("="*60)
        print("🔍 DIAGNOSTICS DU PROJET R8P3-PIPELINE")
        print("="*60)
        self.check_indexes()
        self.check_ollama()
        print("="*60)

if __name__ == "__main__":
    manager = ProjectManager()
    manager.run_diagnostics()