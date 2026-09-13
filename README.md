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
- **Multi-Modal Support** — optical, multispectral, SAR, and cross-modal image pairs
- **GeoTIFF Processing** — full metadata extraction, CRS detection, band inspection, tiling
- **Interactive Map Viewer** — zoom, pan, before/after comparison, overlay, opacity control
- **Evidence Visualization** — bounding boxes, masks, change maps, highlighted regions
- **Confidence Estimation** — calibrated uncertainty with explicit confidence levels
- **Hallucination Control** — model confidence + evidence availability + consistency checks
- **Execution Trace** — step-by-step auditable record of every analysis
- **Benchmark Evaluation** — integrated evaluation on VRSBench, RSVQA, CDVQA
- **Downloadable Reports** — PDF/HTML with full analysis, evidence, and metadata
- **Fallback Mode** — graceful degradation with clearly labelled demo inference

---

## 4. Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- GDAL system library
- (Optional) NVIDIA GPU with CUDA 11.8+

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

SatQuery AI supports four benchmark datasets. See [docs/dataset_config.md](docs/dataset_config.md) for download links and format details.

| Dataset | Purpose | Format |
|---------|---------|--------|
| BigEarthNet | RS image-text adaptation | GeoTIFF + labels |
| VRSBench | Captioning, grounding, VQA | Images + JSON annotations |
| RSVQA | Single-image VQA | Images + QA pairs |
| CDVQA | Bi-temporal change VQA | Image pairs + QA pairs |

```bash
# Place datasets in:
datasets/
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

```bash
# Run evaluation pipelines independently
python -m backend.app.evaluation.run --dataset vrsbench --task vqa
python -m backend.app.evaluation.run --dataset rsvqa --task vqa
python -m backend.app.evaluation.run --dataset cdvqa --task change_vqa
```

Metrics generated per task:

| Task | Metrics |
|------|---------|
| VQA | Accuracy, F1, Exact Match |
| Captioning | BLEU, ROUGE, CIDEr, BERTScore |
| Grounding | IoU, Precision, Recall, mAP |
| Change Detection | IoU, F1, Precision, Recall |
| Change VQA | Accuracy, F1, Semantic Similarity |

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
