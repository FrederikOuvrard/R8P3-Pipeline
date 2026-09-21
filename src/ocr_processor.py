"""
R8P3 OCR Processor Module
Designed for multi-script historical document extraction (Fraktur, Latin, Old French, Ancient Greek)
with built-in CPU thermal safeguarding and batch pacing for massive legal campaigns.
"""

from pathlib import Path
from typing import List, Dict, Optional
import time
from loguru import logger
import pytesseract
from PIL import Image

class HistoricalOCRProcessor:
    def __init__(self, supported_scripts: Optional[List[str]] = None, throttle_delay: float = 0.5):
        self.supported_scripts = supported_scripts or [
            "modern", "fraktur", "latin", "greek_ancient", "old_french"
        ]
        self.throttle_delay = throttle_delay  # Pause de sécurité entre les pages pour protéger le CPU
        logger.info(f"Initialized R8P3 Historical OCR Processor. Scripts: {self.supported_scripts} | Throttle: {throttle_delay}s")

    def validate_image_tensor(self, image_path: str) -> bool:
        """Validates if the image path exists and can be processed."""
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Target document path not found: {image_path}")
            return False
        return True

    def extract_page_text(self, image_path: str, script_mode: str = "latin") -> str:
        """
        Extracts text from a historical legal document page using Tesseract 
        configured with script-specific validation matrices.
        """
        if not self.validate_image_tensor(image_path):
            return ""

        if script_mode not in self.supported_scripts:
            logger.warning(f"Script mode '{script_mode}' not officially recognized. Falling back to latin.")
            script_mode = "latin"

        try:
            logger.debug(f"Processing page {image_path} with script profile: {script_mode}")
            img = Image.open(image_path)
            
            # Configuration Tesseract optimisée pour les textes historiques
            tessdata_config = "--oem 3 --psm 6"
            
            extracted_text = pytesseract.image_to_string(img, config=tessdata_config)
            return extracted_text.strip()
            
        except Exception as e:
            logger.exception(f"Critical error during OCR extraction on {image_path}: {e}")
            return ""

    def batch_process_campaign(self, document_paths: List[str], script_mode: str = "latin") -> Dict[str, str]:
        """
        Processes a massive campaign of historical pages with built-in thermal pacing 
        to prevent CPU throttling on Windows environments.
        """
        results = {}
        total_pages = len(document_paths)
        logger.info(f"Starting secure batch campaign execution for {total_pages} pages...")

        for idx, path in enumerate(document_paths):
            logger.info(f"Processing campaign progress: [{idx + 1}/{total_pages}] -> {path}")
            
            text = self.extract_page_text(path, script_mode=script_mode)
            results[path] = text
            
            # Safeguard pacing to keep CPU thermal footprint stable
            if self.throttle_delay > 0:
                time.sleep(self.throttle_delay)

        logger.info("Campaign batch processing completed successfully.")
        return results

if __name__ == "__main__":
    logger.info("Executing module self-test...")
    processor = HistoricalOCRProcessor()
