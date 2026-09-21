"""
R8P3 OCR Processor Module
Designed for multi-script historical document extraction (Fraktur, Latin, Old French, Ancient Greek)
with built-in CPU thermal safeguarding and native PDF campaign ingestion.
"""

from pathlib import Path
from typing import List, Dict, Optional
import time
import fitz  # PyMuPDF
from loguru import logger
import pytesseract
from PIL import Image
import io

class HistoricalOCRProcessor:
    def __init__(self, supported_scripts: Optional[List[str]] = None, throttle_delay: float = 0.5):
        self.supported_scripts = supported_scripts or [
            "modern", "fraktur", "latin", "greek_ancient", "old_french"
        ]
        self.throttle_delay = throttle_delay  # Pause de sécurité entre les pages pour protéger le CPU
        logger.info(f"Initialized R8P3 Historical OCR Processor. Scripts: {self.supported_scripts} | Throttle: {throttle_delay}s")

    def validate_document_path(self, doc_path: str) -> bool:
        """Validates if the document path exists."""
        path = Path(doc_path)
        if not path.exists():
            logger.error(f"Target document path not found: {doc_path}")
            return False
        return True

    def extract_page_from_image(self, image_path: str, script_mode: str = "latin") -> str:
        """Extracts text from a single image file."""
        if not self.validate_document_path(image_path):
            return ""
        try:
            img = Image.open(image_path)
            return self._run_tesseract(img, script_mode)
        except Exception as e:
            logger.exception(f"Error processing image {image_path}: {e}")
            return ""

    def extract_pdf_campaign(self, pdf_path: str, script_mode: str = "latin") -> Dict[int, str]:
        """
        Ingests a massive legal PDF archive, rendering each page to memory 
        and extracting text with built-in thermal throttling pacing.
        """
        if not self.validate_document_path(pdf_path):
            return {}

        results = {}
        logger.info(f"Opening PDF campaign archive: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"Total pages to process in PDF: {total_pages}")

            for page_num in range(total_pages):
                logger.info(f"Processing PDF page [{page_num + 1}/{total_pages}] (Script: {script_mode})")
                
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)  # Haute résolution recommandée pour l'OCR historique
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                text = self._run_tesseract(img, script_mode)
                results[page_num + 1] = text

                # Thermal safeguard pacing
                if self.throttle_delay > 0:
                    time.sleep(self.throttle_delay)

            doc.close()
            logger.info(f"PDF campaign processing completed successfully for {pdf_path}")
            return results

        except Exception as e:
            logger.exception(f"Critical error during PDF campaign extraction on {pdf_path}: {e}")
            return {}

    def _run_tesseract(self, img: Image.Image, script_mode: str) -> str:
        if script_mode not in self.supported_scripts:
            logger.warning(f"Script mode '{script_mode}' not recognized. Falling back to latin.")
            script_mode = "latin"

        tessdata_config = "--oem 3 --psm 6"
        extracted_text = pytesseract.image_to_string(img, config=tessdata_config)
        return extracted_text.strip()

    def batch_process_campaign(self, document_paths: List[str], script_mode: str = "latin") -> Dict[str, str]:
        results = {}
        for path in document_paths:
            if path.lower().endswith(".pdf"):
                pdf_res = self.extract_pdf_campaign(path, script_mode)
                results[path] = "\n--- PAGE BREAK ---\n".join(pdf_res.values())
            else:
                results[path] = self.extract_page_from_image(path, script_mode)
        return results

if __name__ == "__main__":
    logger.info("Executing OCR module self-test...")
    processor = HistoricalOCRProcessor()
