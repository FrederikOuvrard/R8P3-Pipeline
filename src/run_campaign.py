"""
Campaign Execution Script for R8P3-Pipeline - Codex v42 Legal Corpus
Targets the 20 heavy PDF files from the condensation corpus.
"""

from pathlib import Path
from src.ocr_processor import HistoricalOCRProcessor

def execute_legal_campaign():
    corpus_dir = Path(r"D:\TheseIA\work\these-cowork\reports\fable-lots\codex-v42-condensation-corpus-20260912-r1\preuve-lot31-709-736\source-pdf")

    pdf_files = sorted(list(corpus_dir.glob("*.pdf")))

    print(f"=== CAMPAGNE OCR : CODEX V42 CORPUS JURIDIQUE (R8P3) ===")
    print(f"Dossier cible : {corpus_dir}")
    print(f"Nombre de fichiers PDF détectés : {len(pdf_files)}")

    processor = HistoricalOCRProcessor(
        supported_scripts=["latin", "fraktur"],
        throttle_delay=0.3
    )

    success_count = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[Traitement {i}/{len(pdf_files)}] Fichier : {pdf_path.name}...")
        try:
            extracted_text = processor.extract_pdf_campaign(str(pdf_path), script_mode="latin")
            print(f"[Succès] {len(extracted_text)} caractères extraits de {pdf_path.name}.")
            success_count += 1
        except Exception as e:
            print(f"[Erreur] Échec sur {pdf_path.name}: {e}")

    print(f"\n=== BILAN DE LA CAMPAGNE DU CORPUS ===")
    print(f"Fichiers traités avec succès : {success_count} / {len(pdf_files)}")

if __name__ == "__main__":
    execute_legal_campaign()
