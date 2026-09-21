"""
Unit tests for R8P3 Pipeline components (OCR Processor & Hybrid Search)
Ensures rigorous verification for historical legal text extraction and indexing.
"""

import pytest
from pathlib import Path
from src.ocr_processor import HistoricalOCRProcessor
from src.hybrid_search import HybridSearchEngine

def test_ocr_processor_initialization():
    processor = HistoricalOCRProcessor()
    assert "latin" in processor.supported_scripts
    assert "fraktur" in processor.supported_scripts
    assert processor.throttle_delay == 0.3

def test_ocr_invalid_path():
    processor = HistoricalOCRProcessor()
    # A non-existent document path must return an empty string gracefully
    result = processor.extract_page_text("non_existent_historical_page_999.png", script_mode="latin")
    assert result == ""

    def test_ocr_script_fallback():
        processor = HistoricalOCRProcessor(supported_scripts=["latin"])
        # Test que l'argument est bien pris en compte
        assert "latin" in processor.supported_scripts

def test_hybrid_search_initialization(tmp_path):
    index_dir = tmp_path / "tantivy_index"
    engine = HybridSearchEngine(tantivy_index_path=str(index_dir))
    assert engine.initialize_indices() is True
    assert index_dir.exists()

def test_hybrid_search_stubs():
    engine = HybridSearchEngine(tantivy_index_path="./dummy_index")
    assert engine.search_exact("fiducie-sûreté") == []
    assert engine.search_semantic([0.1, 0.2, 0.3]) == []
    assert engine.hybrid_fusion_ranking([], []) == []

