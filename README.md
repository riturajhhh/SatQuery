# SatQuery AI

## Agentic Vision-Language Assistant for Remote-Sensing Imagery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An agentic, query-driven orchestration framework for remote-sensing vision-language analysis that dynamically selects and combines specialist models across single-image, bi-temporal, and cross-modal optical-SAR workflows while producing evidence and an auditable execution trace.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Features](#3-features)
4. [Installation](#4-installation)
5. [Dataset Setup](#5-dataset-setup)
6. [Model Setup](#6-model-setup)
7. [Training & Adaptation](#7-training--adaptation)
8. [Evaluation](#8-evaluation)
9. [Running Locally](#9-running-locally)
10. [API Documentation](#10-api-documentation)
11. [Screenshots](#11-screenshots)
12. [Example Queries](#12-example-queries)
13. [Limitations](#13-limitations)
14. [Future Work](#14-future-work)

---

## 1. Project Overview

**SatQuery AI** is a modular, evidence-grounded remote-sensing analysis platform that allows users to ask natural-language questions about satellite imagery. The system automatically determines the user's requested task, validates the uploaded imagery, selects the appropriate specialist model/tool, executes the workflow, combines results, and returns textual + visual evidence.

### Supported Workflows

| Workflow | Input | Output |
|----------|-------|--------|
| **Single-Image VQA** | 1 image + question | Answer + confidence + evidence |
| **Image Captioning** | 1 image | Scene description + land-cover interpretation |
| **Visual Grounding** | 1 image + text phrase | Bounding box / mask + visualization |
| **Change Detection** | 2 bi-temporal images | Change map + statistics + visualization |
| **Change VQA** | 2 bi-temporal images + question | Change answer + evidence |
| **Optical-SAR Analysis** | 1 optical + 1 SAR image | Fused observations + visual evidence |

### Key Principles

- **No foundation models trained from scratch** — uses pretrained models with LoRA/PEFT adaptation
- **Modular model registry** — every model is swappable without rewriting the application
- **Evidence-grounded responses** — spatial evidence accompanies every textual answer
- **Confidence estimation** — explicit uncertainty flags (HIGH / MEDIUM / LOW / UNCERTAIN)
- **Hallucination control** — multi-layer validation prevents fabricated observations
- **Auditable execution trace** — machine-readable trace for every analysis

---

## 2. Architecture

```
USER
 ↓
WEB GUI (React + TypeScript + Tailwind CSS)
 ↓
IMAGE + QUERY VALIDATION
 ↓
AGENTIC CONTROLLER
 ↓
QUERY/TASK CLASSIFICATION
 ↓
INPUT/MODALITY ANALYSIS
 ↓
MODEL/TOOL REGISTRY
 ↓
SPECIALIST WORKFLOW
 ↓
EVIDENCE EXTRACTION
 ↓
RESULT VALIDATION
 ↓
RESPONSE GENERATION
 ↓
ANSWER + VISUAL EVIDENCE + CONFIDENCE + EXECUTION TRACE
 ↓
DOWNLOADABLE REPORT
```

### Component Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | React 18 + TypeScript + Tailwind CSS | Professional dashboard with image viewer |
| Backend | Python 3.10+ + FastAPI | REST API, async processing, model orchestration |
| Database | SQLite (→ PostgreSQL) | Analysis history, metadata, execution traces |
| ML Runtime | PyTorch + HuggingFace Transformers | Model inference with GPU/CPU support |
| Geospatial | Rasterio + GDAL + Shapely | GeoTIFF processing, CRS, reprojection |
| Map Viewer | Leaflet / OpenLayers | Interactive satellite imagery visualization |
| Reports | WeasyPrint / ReportLab | PDF/HTML report generation |

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

---

## 3. Features

- **Agentic Task Routing** — automatic detection of VQA, captioning, grounding, change, or SAR analysis intent
- **Adaptive Dual Foundation VLMs (Qwen2-VL-2B + Microsoft Florence-2)** — intelligent arbiter (`adaptive-rs-captioner`) querying both state-of-the-art vision-language models to select whichever outputs richer, non-repetitive descriptions with zero mode collapse:
  - **Qwen2-VL-2B-Instruct**: Native dynamic resolution preserving native satellite aspect ratios, fine-grained multi-paragraph land-use narratives, and zero-shot VQA.
  - **Microsoft Florence-2-base / large**: Granular `<MORE_DETAILED_CAPTION>` and sub-pixel spatial coordinate grounding (`<loc_0>` to `<loc_999>`).
- **Sentinel-2 Optical Specialist Model (`Sentinel2OpticalNet`)** — deep spectral-spatial attention neural network fine-tuned on **BigEarthNet-S2 / EuroSAT** multispectral imagery (Blue, Green, Red, NIR) for 10-class land-cover classification, exact radiometric geophysical indices (mean $NDVI$, $NDWI$, $NDBI$), Color-Infrared (CIR) false-color composite generation, and vegetation vigor evaluation.
- **Sentinel-1 SAR Specialist Model (`Sentinel1SARNet`)** — dedicated deep polarimetric neural network fine-tuned on **BigEarthNet Sentinel-1 C-Band SAR** dual-polarization ($VV / VH$) data for calibrated decibel backscatter quantification ($\sigma^0$), dihedral double-bounce building/ship localization, specular water absorption mapping, and radar-specific question answering (SAR-VQA).
- **Standalone & Cross-Modal Optical-SAR Tri-Stream Engine** — fully decoupled handling for Standalone Optical, Standalone SAR, and Optical + SAR Cross-Modal Fusion with cloud penetration and cross-sensor verification.
- **Curated Workflow Query Presets** — categorized, highly relatable queries across 🌿 Single Optical, 📡 Single SAR, ⚡ Optical + SAR, 🔄 Bi-Temporal Change, and 🇮🇳 ISRO Missions.
- **HSPD-Change Engine** — Hierarchical Structural-Phenological Decoupled change detection with Relative Radiometric Normalization (RRN), gradient tensor structural dissimilarity, and discrete building counter delta tracking
- **AG-MFD Optical-SAR Engine** — Adaptive Geophysical Multi-Scale Frequency Decomposition with Lee local-variance speckle filter, Dynamic Spectral Cloud Index (DSCI), 3-class polarimetric backscatter decomposition, and Laplacian pyramid cross-fusion
- **Multi-Modal Support** — optical, multispectral, SAR, and cross-modal image pairs
- **GeoTIFF Processing** — full metadata extraction, CRS detection, band inspection, tiling
- **Interactive Map Viewer** — zoom, pan, split-slider before/after & optical/radar comparison, overlay, opacity control
- **Evidence Visualization** — bounding boxes, masks, change maps, highlighted regions
- **Confidence Estimation** — calibrated uncertainty with explicit confidence levels
- **Hallucination Control** — model confidence + evidence availability + consistency checks
- **Execution Trace** — step-by-step auditable record of every analysis
- **Benchmark Evaluation** — integrated evaluation on LEVIR-CD, VRSBench, RSVQA, CDVQA
- **Downloadable Reports** — PDF/HTML with full analysis, evidence, and metadata
- **Fallback Mode** — graceful degradation with clearly labelled demo inference

---

## 4. Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- GDAL system library
- (Optional) NVIDIA GPU with CUDA 11.8+ (Florence-2 / BIT / BLIP-2)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Environment Configuration

```bash
cp .env.example .env
# Edit .env with your configuration
```

---

## 5. Dataset Setup

SatQuery AI integrates five benchmark remote-sensing datasets. See [docs/dataset_config.md](docs/dataset_config.md) for download links and format details.

| Dataset | Purpose | Tasks | Format | Size |
|---------|---------|-------|--------|------|
| **LEVIR-CD** | VHR Bi-Temporal Building Change Detection | Building Change Detection, Change VQA | 0.5m Optical pairs + binary masks | ~4GB |
| **BigEarthNet** | RS image-text adaptation | Multi-label classification | GeoTIFF + labels | ~66GB (S2) |
| **VRSBench** | Captioning, grounding, VQA | Caption, Ground, VQA | Images + JSON annotations | ~5GB |
| **RSVQA** | Single-image VQA | VQA | Images + QA pairs | ~1-15GB |
| **CDVQA** | Bi-temporal change VQA | Change VQA | Image pairs + QA pairs | ~3GB |

```bash
# Ingest or generate LEVIR-CD sample pairs:
python scripts/download_levir_cd.py --generate-samples

# Place datasets in:
datasets/
├── levir_cd/
├── bigearthnet_txt/
├── vrsbench/
├── rsvqa/
└── cdvqa/
```

---

## 6. Model Setup

Models are registered in the model registry and loaded on demand. See [docs/model_registry.md](docs/model_registry.md).

```bash
# Place model weights in:
models/
├── vqa/
├── captioning/
├── grounding/
├── change_detection/
├── change_vqa/
└── optical_sar/
```

Pretrained weights are downloaded automatically on first use. LoRA adapters are stored separately from base model weights.

---

## 7. Training & Adaptation

At least one VLM component is adapted to remote sensing using BigEarthNet with LoRA/PEFT. See training documentation in `scripts/` for:

- Base model selection
- Dataset preprocessing
- Training configuration (epochs, learning rate, adapter config)
- Train/validation split
- Evaluation results

---

## 8. Evaluation

SatQuery AI includes automated benchmark evaluators for all core tasks:

```bash
# Run benchmark evaluation runner (from backend/ directory or root with PYTHONPATH):
python -m app.evaluation.runner --benchmark levir_cd
python -m app.evaluation.runner --benchmark vrsbench
python -m app.evaluation.runner --benchmark rsvqa
python -m app.evaluation.runner --benchmark cdvqa
python -m app.evaluation.runner --benchmark all
```

Metrics generated per task:

| Task / Benchmark | Primary Metrics |
|---|---|
| **LEVIR-CD** (Building Change Detection) | Change IoU, Recall, Precision, F1-Score, Overall Accuracy (OA), Cohen's Kappa |
| **VRSBench** (VQA, Captioning, Grounding) | Accuracy, BLEU-4, ROUGE-L, CIDEr, Box IoU, Precision@0.5 |
| **RSVQA** (High-Resolution VQA) | Accuracy, Macro F1, Per-Category Accuracy |
| **CDVQA** (Bi-Temporal Change VQA) | Answer Accuracy, F1, Semantic Similarity |
| **Optical-SAR / Cross-Modal** | Alignment IoU, Structural Similarity (SSIM), Modal Consistency Score |

---

## 9. Running Locally

### Start Backend

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
cd frontend
npm run dev
```

### Docker (optional)

```bash
docker-compose up --build
```

Access the application at `http://localhost:5173`.

---

## 10. API Documentation

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload satellite images |
| `/api/analyze` | POST | Submit analysis query |
| `/api/analysis/{id}` | GET | Retrieve analysis results |
| `/api/analysis/{id}/trace` | GET | Get execution trace |
| `/api/analysis/{id}/evidence` | GET | Get visual evidence |
| `/api/analysis/{id}/report` | GET | Download PDF/HTML report |
| `/api/models` | GET | List available models |
| `/api/health` | GET | Health check |

See [docs/api_design.md](docs/api_design.md) for full specification.

---

## 11. Screenshots

*Screenshots will be added after Phase 14 (Final UI/UX).*

---

## 12. Example Queries

### Single Image VQA
> "Describe the land-cover and major objects visible in this image."

### Visual Grounding
> "Highlight the water body referred to in the query."

### Bi-Temporal Change
> "What changed between these two dates, and where did the change occur?"

### Optical-SAR Fusion
> "Use the optical and SAR images together to identify built-up and water-covered regions."

### Urban Change Analysis
> "Has the built-up area increased, decreased, or remained unchanged?"

---

## 13. Limitations

> [!IMPORTANT]
> Please read these limitations carefully before using SatQuery AI.

- Satellite imagery is **not necessarily real-time**; results reflect the latest available observation
- Results depend on **image quality** — cloud cover may affect optical imagery
- SAR interpretation is complex and may produce ambiguous results
- Co-registration errors between image pairs can create false changes
- Vision-language models **can hallucinate** — always verify critical findings
- Benchmark performance does **not guarantee** real-world performance
- Geographical and domain distribution may cause **dataset bias**
- Confidence scores indicate **relative reliability**, not absolute correctness
- The system uses "latest available observation" or "near-real-time where supported by the imagery provider"

**SatQuery AI never claims:**
- "100% accurate"
- "Real-time satellite monitoring"

---

## 14. Future Work

- Multi-GPU distributed inference
- Streaming analysis for very large images
- Integration with cloud-native geospatial platforms (STAC, COG)
- Support for hyperspectral imagery
- Temporal series analysis (>2 time steps)
- Active learning for domain adaptation
- User annotation and feedback loop
- PostgreSQL migration for production deployment
- Kubernetes deployment with auto-scaling
- Integration with real-time satellite data providers

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Research Positioning

SatQuery AI does not claim to have invented VQA, image captioning, visual grounding, change detection, SAR processing, optical-SAR fusion, or remote-sensing VLMs.

> **Contribution:** An agentic, query-driven orchestration framework for remote-sensing vision-language analysis that dynamically selects and combines specialist models across single-image, bi-temporal, and cross-modal optical-SAR workflows while producing evidence and an auditable execution trace.
