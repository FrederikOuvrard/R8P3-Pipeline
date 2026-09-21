\"\"\"
R8P3 Main Execution Pipeline (Production Grade)
Orchestrates multi-script historical OCR processing with thermal safeguarding 
and hybrid indexing (Tantivy + Qdrant) driven by config/settings.yaml.
\"\"\"

import sys
from pathlib import Path
import yaml
from loguru import logger

from src.ocr_processor import HistoricalOCRProcessor
from src.hybrid_search import HybridSearchEngine

def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Configuration file not found at {config_path}. Falling back to default parameters.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_pipeline_demo():
    # Load configuration
    config = load_config()
    log_level = config.get("logging", {}).get("level", "INFO")

    # Configure Loguru observability
    logger.remove()
    logger.add(sys.stdout, level=log_level)

    logger.info("=== Starting R8P3 Pipeline Orchestration (Production Grade) ===")

    ocr_config = config.get("ocr", {})
    search_config = config.get("search", {})

    # 1. Initialize OCR Processor with YAML configuration
    ocr_processor = HistoricalOCRProcessor(
        supported_scripts=ocr_config.get("supported_scripts"),
        throttle_delay=ocr_config.get("throttle_delay_seconds", 0.3)
    )

    # 2. Initialize Hybrid Search Engine from YAML configuration
    tantivy_path = search_config.get("tantivy", {}).get("index_path", "./data/tantivy_index")
    qdrant_host = search_config.get("qdrant", {}).get("host", "localhost")
    qdrant_port = search_config.get("qdrant", {}).get("port", 6333)

    search_engine = HybridSearchEngine(
        tantivy_index_path=tantivy_path,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port
    )
    
    if not search_engine.initialize_indices():
        logger.error("Failed to initialize hybrid search indices. Aborting execution.")
        return

    # 3. Simulate batch historical document campaign execution
    sample_campaign = [
        "data/raw/historical_deed_page_01.png",
        "data/raw/historical_deed_page_02.png"
    ]
    
    default_script = ocr_config.get("default_script", "latin")
    logger.info(f"Triggering secure batch OCR campaign for {len(sample_campaign)} pages using script profile: {default_script}...")
    
    ocr_results = ocr_processor.batch_process_campaign(
        sample_campaign, 
        script_mode=default_script
    )

    # 4. Execute Hybrid Search and Fusion Ranking demo
    logger.info("Executing sample hybrid search queries on historical corpus...")
    exact_matches = search_engine.search_exact("fiducie-sûreté", limit=5)
    semantic_matches = search_engine.search_semantic([0.15, 0.22, 0.38], limit=5)
    
    alpha_val = search_config.get("fusion", {}).get("alpha", 0.5)
    final_ranking = search_engine.hybrid_fusion_ranking(exact_matches, semantic_matches, alpha=alpha_val)

    logger.info("=== R8P3 Pipeline Production Execution Completed Successfully ===")

if __name__ == "__main__":
    run_pipeline_demo()
