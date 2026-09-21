import os
import pytesseract
os.environ["TESSDATA_PREFIX"] = r"D:\R8P3-Pipeline\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\PDF24\tesseract\tesseract.exe"
"""
R8P3 Real Campaign Execution Script
Ingests a directory of historical legal documents (PDFs/Scans), runs multi-script OCR
with thermal safeguards, and indexes the resulting corpus into Tantivy and Qdrant.
"""

import sys
from pathlib import Path
import yaml
from loguru import logger

from src.ocr_processor import HistoricalOCRProcessor
from src.hybrid_search import HybridSearchEngine

def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    logger.info("=== R8P3 Historical Campaign Runner ===")

    config = load_config()
    ocr_cfg = config.get("ocr", {})
    search_cfg = config.get("search", {})

    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = (".pdf", ".png", ".jpg", ".jpeg", ".tiff")
    target_files = [str(p) for p in raw_data_dir.iterdir() if p.suffix.lower() in supported_extensions]

    if not target_files:
        logger.warning(f"No target PDF or image documents found in '{raw_data_dir.absolute()}' Folder.")
        target_files = ["data/raw/sample_historical_archive.pdf"]
        logger.info(f"Using simulated document reference for demonstration: {target_files[0]}")

    # 1. Initialize OCR Processor with thermal throttling
    ocr_processor = HistoricalOCRProcessor(
        supported_scripts=ocr_cfg.get("supported_scripts"),
        throttle_delay=ocr_cfg.get("throttle_delay_seconds", 0.3)
    )

    # 2. Initialize Hybrid Search Engine
    tantivy_path = search_cfg.get("tantivy", {}).get("index_path", "./data/tantivy_index")
    qdrant_col = search_cfg.get("qdrant", {}).get("collection_name", "historical_legal_corpus")

    search_engine = HybridSearchEngine(
        tantivy_index_path=tantivy_path,
        qdrant_local_path="./data/qdrant_storage",
        collection_name=qdrant_col
    )

    if not search_engine.initialize_indices():
        logger.error("Failed to initialize search indices. Campaign aborted.")
        return

    # 3. Execute campaign OCR batch extraction
    default_script = ocr_cfg.get("default_script", "latin")
    logger.info(f"Starting batch campaign extraction with script profile: '{default_script}'...")

    extraction_results = ocr_processor.batch_process_campaign(target_files, script_mode=default_script)

    # 4. Indexation automatique des résultats dans les moteurs
    logger.info("Indexing extracted documents into Tantivy and Qdrant...")
    for doc_path, text in extraction_results.items():
        doc_name = Path(doc_path).name
        char_count = len(text)
        
        # Indexation textuelle exacte
        search_engine.index_document(doc_name, text)
        logger.info(f"Indexed: {doc_name} | Chars: {char_count}")

    logger.info("=== R8P3 Campaign Execution & Indexing Finished Successfully ===")

if __name__ == "__main__":
    main()
