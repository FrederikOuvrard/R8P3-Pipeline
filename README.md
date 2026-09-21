# R8P3-Pipeline: Scalable Multilingual OCR & Hybrid RAG for Historical Computable Law

[![R8P3 CI Pipeline](https://github.com/FrederikOuvrard/R8P3-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/FrederikOuvrard/R8P3-Pipeline/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)

## 📌 Project Overview
R8P3 is a foundational open-source pipeline designed to bridge a critical gap in AI safety and legal informatics. It provides the digital infrastructure necessary to transform massive, complex historical legal archives (featuring modern print, Fraktur, Latin, Ancient Greek, and Old French) into computable data.

By leveraging advanced multi-script OCR and highly scalable hybrid vector-search architectures, R8P3 enables the extraction of systemic liability models and risk governance frameworks built upon centuries of legal reasoning.

---

## ⚠️ The Problem (Vulnerability)
Currently, critical historical legal frameworks detailing systemic risk management—such as **collateral agreements** and **insolvency schemes**—are virtually invisible to modern AI systems. The processing of these complex historical corpora is severely hindered by:
* Inefficient and fragmented OCR tools unable to handle multi-script historical typography.
* Extreme hardware constraints (thermal throttling) when scaling local pipelines to millions of pages.
* A lack of optimized vectorization architectures capable of massive-scale RAG (Retrieval-Augmented Generation) for historical law.

---

## 🚀 The R8P3 Solution (Base Technology)
R8P3 is not an application; it is a **scalable base technology** for the open-source ecosystem. The pipeline integrates:
1. **Multi-script OCR**: Robust validation matrices for historical scripts (Fraktur, Latin, Old French, Ancient Greek).
2. **Hybrid Vector Search**: Utilizing **Tantivy** (exact-match text search) and **Qdrant** (semantic vectorization) for high-performance indexing and retrieval.
3. **Hardware-Safe Execution**: Built-in runtime pacing to prevent CPU thermal throttling during massive document campaigns.

---

## ⚙️ Installation & Environment Setup

Clone the repository and set up a local virtual environment:

\\\ash
git clone https://github.com/FrederikOuvrard/R8P3-Pipeline.git
cd R8P3-Pipeline
py -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
\\\

---

## 💡 Quick Start & Usage

### 1. Configure Settings
Review and adjust your pipeline parameters in config/settings.yaml:
\\\yaml
ocr:
  supported_scripts: ["modern", "fraktur", "latin", "greek_ancient", "old_french"]
  fallback_engine: "tesseract"
\\\

### 2. Run Campaign Extractor
\\\ash
python src/run_campaign.py
\\\

---

## 📜 License
Distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE for more information.
