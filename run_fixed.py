import os
import sys
from pathlib import Path

# 1. Forcer le chemin absolu de Tesseract pour pytesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\PDF24\tesseract\tesseract.exe"

# 2. S'assurer que le dossier 'src' est bien dans le path Python
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.ocr_processor import HistoricalOCRProcessor
    print("Moteur OCR chargé avec succès.")
    
    # Initialisation et lancement de la campagne
    processor = HistoricalOCRProcessor()
    print("Processeur initialisé. Lancement de la campagne...")
    
    # Ajustez ici selon la logique de votre run_campaign.py d'origine
    # (Traitement des 20 fichiers de la campagne)
    
except Exception as e:
    print(f"Erreur critique lors de l'exécution : {e}")
