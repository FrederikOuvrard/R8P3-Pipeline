\"\"\"
Campaign Execution Script for R8P3-Pipeline
Processes legal PDF corpora from the local workspace under controlled thermal/throttled conditions.
\"\"\"

import os
from pathlib import Path
from src.ocr_processor import HistoricalOCRProcessor

def execute_legal_campaign():
    # Cible le répertoire exact où se trouvent les PDF juridiques Hulot et autres
    corpus_dir = Path(r"D:\TheseIA\work\these-cowork\outputs")
    
    if not corpus_dir.exists():
        # Fallback de secours si le chemin varie
        corpus_dir = Path(r"D:\TheseIA")
        print(f"[Avertissement] Dossier principal introuvable. Recherche globale dans {corpus_dir}")
        pdf_files = list(corpus_dir.glob("**/*.pdf"))
    else:
        pdf_files = list(corpus_dir.glob("**/*.pdf"))

    print(f"=== LANCEMENT DE LA CAMPAGNE OCR R8P3 ===")
    print(f"Dossier cible : {corpus_dir}")
    print(f"Nombre de fichiers PDF détectés : {len(pdf_files)}")

    # Initialisation du processeur OCR (avec support des scripts historiques)
    processor = HistoricalOCRProcessor(
        supported_scripts=["latin", "fraktur"],
        throttle_delay=0.3
    )

    success_count = 0
    for pdf_path in pdf_files[:20]:  # Limitation aux 20 premiers fichiers de la campagne
        print(f"\n[Traitement] En cours : {pdf_path.name}...")
        try:
            # Extraction via le pipeline OCR sécurisé
            extracted_text = processor.extract_pdf_campaign(str(pdf_path), script_mode="latin")
            print(f"[Succès] {len(extracted_text)} caractères extraits de {pdf_path.name}.")
            success_count += 1
        except Exception as e:
            print(f"[Erreur] Échec sur {pdf_path.name}: {e}")

    print(f"\n=== BILAN DE LA CAMPAGNE ===")
    print(f"Fichiers traités avec succès : {success_count} / {min(20, len(pdf_files))}")

if __name__ == "__main__":
    execute_legal_campaign()
