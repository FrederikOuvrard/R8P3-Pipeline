# R8P3-Pipeline: Technical Architecture & Design

## 🏛️ System Overview
R8P3 is engineered as a production-grade, modular open-source pipeline designed to process massive historical legal archives spanning multiple typography systems (Modern, Fraktur, Latin, Ancient Greek, Old French). 

## ⚙️ Core Architectural Components

### 1. Resilient Multi-Script OCR Engine (src/ocr_processor.py)
* **Script Profiles**: Native validation matrices supporting historical legal variations.
* **Thermal Safeguard**: Built-in execution pacing (	hrottle_delay) designed to stabilize CPU thermal footprints during massive batch campaigns on local hardware environments.
* **Error Handling**: Graceful fallback mechanisms preventing pipeline interruptions on corrupted scans or unreadable folios.

### 2. Hybrid Search Engine (src/hybrid_search.py)
* **Tantivy**: High-performance full-text search index dedicated to exact-match legal terminology, citations, and historical statute identifiers (e.g., *quasi-contrats*, *fiducies-sûretés*).
* **Qdrant**: Scalable vector search database capturing semantic conceptual continuity across centuries of jurisprudence.
* **Reciprocal Rank Fusion (RRF)**: Advanced re-ranking algorithm balancing exact lexical matching with deep semantic vector representations.

### 3. Centralized Configuration & CI/CD
* **Declarative Settings**: Managed entirely via config/settings.yaml for complete environment decoupling.
* **Automated Assurance**: Continuous integration pipeline running comprehensive pytest suites via GitHub Actions on every commit.
