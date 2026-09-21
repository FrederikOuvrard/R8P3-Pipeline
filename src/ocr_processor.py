"""
R8P3 Historical OCR Processor
Handles multi-script OCR extraction with dynamic Tesseract path resolution.
"""

import os
from pathlib import Path
from loguru import logger
import fitz  # PyMuPDF
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pytesseract

# Forcer le chemin de Tesseract et le dossier tessdata local du projet
os.environ["TESSDATA_PREFIX"] = str(Path(__file__).resolve().parent.parent)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\PDF24\tesseract\tesseract.exe"

class HistoricalOCRProcessor:
    def __init__(self, supported_scripts: list = None, throttle_delay: float = 0.3):
        self.supported_scripts = supported_scripts or ["latin", "fraktur", "greek", "old_french"]
        self.throttle_delay = throttle_delay
        logger.info(f"Initialized HistoricalOCRProcessor | TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX')}")

    def extract_pdf_campaign(self, pdf_path: str, script_mode: str = "latin") -> str:
        full_text = []
        path = Path(pdf_path)
        
        if not path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return ""

        try:
            logger.info(f"Opening PDF campaign archive: {path}")
            doc = fitz.open(str(path))
            total_pages = len(doc)
            logger.info(f"Total pages to process in PDF: {total_pages}")

            for page_num in range(total_pages):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=300)
                img_path = Path(f"temp_page_{page_num}.png")
                pix.save(str(img_path))

                with Image.open(img_path) as img:
                    text = self._run_tesseract(img, script_mode)
                    full_text.append(text)

                if img_path.exists():
                    img_path.unlink()

            doc.close()
            return "\n".join(full_text)

        except Exception as e:
            logger.exception(f"Critical error during PDF campaign extraction on {pdf_path}: {e}")
            return ""

    def _run_tesseract(self, img: Image.Image, script_mode: str) -> str:
        lang_map = {
            "latin": "lat+fra",
            "fraktur": "deu-frak+lat",
            "greek": "grc+lat",
            "old_french": "fra+lat"
        }
        lang = lang_map.get(script_mode, "lat")
        tessdata_config = "--oem 3 --psm 6"
        
        try:
            return pytesseract.image_to_string(img, lang=lang, config=tessdata_config)
        except Exception:
            return pytesseract.image_to_string(img, config=tessdata_config)

    def batch_process_campaign(self, file_paths: list, script_mode: str = "latin") -> dict:
        results = {}
        for fp in file_paths:
            text = self.extract_pdf_campaign(fp, script_mode=script_mode)
            if text:
                results[fp] = text
        return results
