\"\"\"
R8P3 Main Execution Pipeline
Orchestrates multi-script historical OCR processing with thermal safeguarding 
and hybrid indexing (Tantivy + Qdrant) for computable legal archives.
\"\"\"

import sys
from pathlib import Path
from loguru import logger

from src.ocr_processor import HistoricalOCRProcessor
from src.hybrid_search import HybridSearchEngine

# Configure Loguru for professional observability
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{msg}</level>")

def run_pipeline_demo():
    logger.info("=== Starting R8P3 Pipeline Orchestration Demo ===")

    # 1. Initialize OCR Processor with CPU thermal pacing
    ocr_processor = HistoricalOCRProcessor(
        supported_scripts=["modern", "fraktur", "latin", "greek_ancient", "old_french"],
        throttle_delay=0.3
    )

    # 2. Initialize Hybrid Search Engine (Tantivy + Qdrant)
    index_path = "./data/tantivy_index"
    search_engine = HybridSearchEngine(
        tantivy_index_path=index_path,
        qdrant_host="localhost",
        qdrant_port=6333
    )
    
    if not search_engine.initialize_indices():
        logger.error("Failed to initialize hybrid search indices. Aborting demo.")
        return

    # 3. Simulate a historical document batch processing campaign
    # (Using dummy paths to demonstrate the secure batch execution flow)
    sample_document_campaign = [
        "data/raw/historical_deed_page_01.png",
        "data/raw/historical_deed_page_02.png"
    ]
    
    logger.info(f"Triggering batch OCR campaign for {len(sample_document_campaign)} legal documents...")
    ocr_results = ocr_processor.batch_process_campaign(
        sample_document_campaign, 
        script_mode="fraktur"
    )

    # 4. Demonstrate Hybrid Search Capabilities on extracted corpus
    logger.info("Executing sample hybrid search queries...")
    exact_matches = search_engine.search_exact("fiducie-sûreté", limit=5)
    semantic_matches = search_engine.search_semantic([0.15, 0.22, 0.38], limit=5)

    # RRF Fusion ranking preview
    final_ranking = search_engine.hybrid_fusion_ranking(exact_matches, semantic_matches)

    logger.info("=== R8P3 Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline_demo()
