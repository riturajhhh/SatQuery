# SatQuery AI — Complete Technical & Architectural System Guide

**Project Title:** SatQuery AI — Agentic Vision-Language Assistant for Remote-Sensing Imagery  
**Purpose:** Comprehensive Technical Reference, System Walkthrough, and Final Year Project Defense Guide  
**Repository:** [https://github.com/riturajhhh/SatQuery](https://github.com/riturajhhh/SatQuery)  

---

## Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [High-Level Architecture & End-to-End Data Flow](#2-high-level-architecture--end-to-end-data-flow)
3. [The Core Subsystems — What Was Built & How It Works](#3-the-core-subsystems--what-was-built--how-it-works)
   - 3.1 [Geospatial Ingestion Engine & ISRO/SAC Readiness](#31-geospatial-ingestion-engine--isrosac-readiness)
   - 3.2 [Agentic Query Router & Task Classifier](#32-agentic-query-router--task-classifier)
   - 3.3 [Dual-Engine Model Registry (Production & Heuristic Fallbacks)](#33-dual-engine-model-registry-production--heuristic-fallbacks)
   - 3.4 [The 6 Core Remote-Sensing Workflows](#34-the-6-core-remote-sensing-workflows)
   - 3.5 [Visual Evidence & Confidence Calibration Engine](#35-visual-evidence--confidence-calibration-engine)
   - 3.6 [Auditable Report Generation (PDF & HTML)](#36-auditable-report-generation-pdf--html)
   - 3.7 [Modern React Dashboard & UI Components](#37-modern-react-dashboard--ui-components)
4. [Step-by-Step Life of a Query (Execution Trace)](#4-step-by-step-life-of-a-query-execution-trace)
5. [Roadmap Completion Summary (Phases 0–14)](#5-roadmap-completion-summary-phases-014)
6. [Codebase Organization & File Guide](#6-codebase-organization--file-guide)
7. [Verification, Testing, and Local Execution](#7-verification-testing-and-local-execution)
8. [Final Year Project Defense & Viva Guide (Q&A)](#8-final-year-project-defense--viva-guide-qa)

---

## 1. Executive Summary & Problem Statement

### 1.1 The Problem
Satellite Earth Observation (EO) data is massive, multi-modal (optical, multispectral, SAR), and geometrically complex. Traditional remote sensing analysis requires specialized GIS software (ArcGIS, QGIS, ENVI) and manual algorithmic tuning. Conversely, mainstream multimodal large language models (like generic GPT-4V or base vision models) are trained predominantly on eye-level internet imagery (COCO, ImageNet). When shown top-down satellite imagery, they suffer from:
- **Spatial hallucinations:** Inventing features not present in the scene.
- **Top-down blindness:** Inability to handle nadir perspectives, extreme scale variations, or multispectral bands beyond RGB.
- **Black-box responses:** Stating an answer with no verifiable pixel-grounded evidence.
- **Fragility with Indian EO data:** Inability to process high-dynamic range (10/11/12-bit) Cartosat optical sensors or calibrated $\sigma^0$ dB RISAT SAR polarimetry.

### 1.2 The Solution: SatQuery AI
SatQuery AI is an **agentic, query-driven, evidence-grounded remote-sensing assistant**. Rather than relying on a single brittle monolithic model, SatQuery AI dynamically routes incoming natural language questions to specialist computer vision algorithms, models, and cross-modal fusion pipelines. Every response is validated through a multi-factor hallucination guard, calibrated with an explicit confidence score (HIGH, MEDIUM, LOW, UNCERTAIN), accompanied by visual spatial evidence (masks, bounding boxes, change maps), and compiled into an auditable report.

---

## 2. High-Level Architecture & End-to-End Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             REACT FRONTEND (PORT 5173)                           │
│  - Drag-and-Drop Image Uploader (Single, Bi-temporal, Optical + SAR)             │
│  - Natural Language Query Input & Categorized Preloaded Demo Chips               │
│  - Hardware-Accelerated Before/After Swipe Slider                                │
│  - Visual Evidence Viewer (Opacity 10%-100%, Zoom Modal)                         │
│  - Execution Trace Timeline (Stage pills, latencies, millisecond breakdown)      │
│  - Model Registry Modal & Analysis History Drawer                                │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ HTTP REST / JSON
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND SERVER (PORT 8000)                       │
│                                                                                  │
│  ┌───────────────────────┐   ┌────────────────────────┐   ┌───────────────────┐  │
│  │   API Routing Layer   │   │  Database Layer        │   │  Static File Host │  │
│  │  /api/upload          │   │  SQLite + SQLAlchemy   │   │  /uploads/        │  │
│  │  /api/analyze         │   │  (Analyses, Files,     │   │  /outputs/        │  │
│  │  /api/analysis/{id}   │   │   Traces, Evidence)    │   │  /reports/        │  │
│  └───────────┬───────────┘   └───────────┬────────────┘   └───────────────────┘  │
│              │                           │                                       │
│              ▼                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                        GEOSPATIAL INGESTION ENGINE                         │  │
│  │  - Rasterio & GDAL GeoTIFF Parser (CRS, Bounding Box, Resolution, Bands)   │  │
│  │  - Modality Detector (Optical, Multispectral, SAR Backscatter)             │  │
│  │  - ISRO/SAC Sensor Normalizer (Cartosat-2S/3 uint16, RISAT-1A EOS-04 dB)   │  │
│  │  - Bi-Temporal Sub-Pixel Alignment & Pair Compatibility Validator          │  │
│  │  - Dynamic 8-bit Contrast-Stretched PNG Thumbnail & Preview Generator      │  │
│  └───────────────────────────────────────┬────────────────────────────────────┘  │
│                                          │                                       │
│                                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                        AGENTIC ORCHESTRATOR & PLANNER                      │  │
│  │  - Natural Language Intent Parser & Regex/Keyword Classification           │  │
│  │  - Sensor & Image Modality Constraints Matcher                             │  │
│  │  - Dynamic Task Router:                                                    │  │
│  │      * Single-Image VQA          * Change Detection (Bi-Temporal)          │  │
│  │      * Scene Captioning          * Change VQA                              │  │
│  │      * Visual Grounding          * Cross-Modal Optical-SAR Analysis        │  │
│  └───────────────────────────────────────┬────────────────────────────────────┘  │
│                                          │                                       │
│                                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                        SPECIALIST MODEL REGISTRY                           │  │
│  │  ┌─────────────────────────────────┐   ┌────────────────────────────────┐  │  │
│  │  │    Production Models (GPU)      │   │   Deterministic CPU Fallback   │  │  │
│  │  │  - BLIP-2 VQA (LoRA BigEarthNet)│   │  - Spectral NDVI/NDWI Indices  │  │  │
│  │  │  - BLIP-2 RS Captioning         │   │  - Otsu Bitemporal Diff Engine │  │  │
│  │  │  - Grounding DINO RS            │   │  - Contour Morphological BBoxes│  │  │
│  │  │  - BIT Change Detection Net     │   │  - SAR Backscatter Despeckle   │  │  │
│  │  └─────────────────────────────────┘   └────────────────────────────────┘  │  │
│  └───────────────────────────────────────┬────────────────────────────────────┘  │
│                                          │                                       │
│                                          ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │               EVIDENCE, CONFIDENCE & AUDIT COMPILATION                     │  │
│  │  - Visual Evidence Synthesis (Colorized Heatmaps, Overlays, Bounding Boxes)│  │
│  │  - Calibrated Confidence Engine (Model Score + Evidence + Sensor Quality)   │  │
│  │  - Multi-Layer Hallucination Guard (Cross-checks claims with spatial data) │  │
│  │  - Execution Trace Builder (Step-by-step millisecond timeline)             │  │
│  │  - ReportLab PDF & Print-Optimized HTML Export Generator                   │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Core Subsystems — What Was Built & How It Works

### 3.1 Geospatial Ingestion Engine & ISRO/SAC Readiness
*Implementation:* `backend/app/geospatial/metadata.py`, `modality.py`, `preview.py`, `isro.py`, `compatibility.py`

1. **Format Handling:** Uses `rasterio` and GDAL to ingest standard GeoTIFF, TIFF, and common raster imagery without crashing.
2. **Metadata Extraction:** Extracts Coordinate Reference System (CRS, e.g. EPSG:4326, UTM Zones 42N–46N for India, Kalianpur 1975), geographic bounding box coordinates, spatial resolution ($x, y$ ground sample distance in meters), number of spectral bands, and data type (`uint8`, `uint16`, `float32`).
3. **Modality Identification:** Automatically determines if an image is:
   - *Optical RGB* (3 bands, visible spectrum)
   - *Multispectral* (4+ bands including Red Edge, NIR, SWIR)
   - *SAR (Synthetic Aperture Radar)* (single/dual polarization, C-band/X-band, backscatter intensity in decibels $\sigma^0$)
4. **ISRO/SAC Compatibility:** Specifically engineered for Indian space sensors:
   - **Cartosat-2S / Cartosat-3:** Handles high dynamic range (10-bit to 12-bit stored in `uint16`) using 2%–98% cumulative percentile clipping to prevent washed-out or black previews.
   - **RISAT-1 / RISAT-1A (EOS-04):** Handles SAR amplitude/intensity and converts linear values to decibel backscatter:
     $$\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\text{amplitude}^2 + 10^{-7})$$
5. **Pair Compatibility Validator:** When a user uploads two images for change detection or Optical-SAR fusion, the engine verifies dimension alignment, CRS matching, and spatial overlap.

---

### 3.2 Agentic Query Router & Task Classifier
*Implementation:* `backend/app/services/agentic_planner.py`

Instead of expecting the user to manually select "VQA Mode" or "Change Mode", the agent analyzes:
- The **linguistic intent** of the natural-language prompt (e.g. *"describe"*, *"where is"*, *"what changed"*, *"count"*, *"radar penetration"*).
- The **cardinality and modality** of the uploaded images (1 optical, 2 temporal opticals, 1 optical + 1 SAR).

#### Routing Logic Matrix:
| Uploaded Input | Query Intent | Routed Specialist Task |
|---|---|---|
| Single Image | "Describe the scene / land-cover" | **Image Captioning** |
| Single Image | "Where is the [object] / locate [phrase]" | **Visual Grounding** |
| Single Image | "What is / how many / is there a..." | **Single-Image VQA** |
| Two Images (Bi-temporal) | "What changed / detect modifications" | **Change Detection** |
| Two Images (Bi-temporal) | "Why did the lake shrink? / was building added?" | **Change VQA** |
| Optical + SAR Pair | "Fuse SAR and optical / penetrate cloud cover" | **Optical-SAR Analysis** |

---

### 3.3 Dual-Engine Model Registry (Production & Heuristic Fallbacks)
*Implementation:* `backend/app/models/registry.py`, `backend/app/models/base.py`

A major architectural pillar of SatQuery AI is its **zero-downtime, fault-tolerant dual-engine design**:
1. **Production Engine (GPU / Deep Learning):**
   - Connects to PyTorch/HuggingFace transformer backbones (BLIP-2, Grounding DINO, BIT Change Net) adapted via LoRA for remote sensing.
2. **Deterministic Heuristic Engine (CPU Fallback):**
   - If CUDA is unavailable, weights are missing, or a system is memory-constrained, the engine smoothly falls back to deterministic geospatial algorithms without failing or throwing an unhandled exception.
   - Computes normalized spectral indices:
     $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}, \quad \text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}, \quad \text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}$$
   - Uses Otsu automatic bi-level thresholding for bitemporal change detection.
   - Uses morphological contour bounding algorithms for visual grounding.

---

### 3.4 The 6 Core Remote-Sensing Workflows

1. **Single-Image VQA (`vqa.py`):**
   Answers specific questions on land-cover types, urban infrastructure, water bodies, roads, and structures with exact confidence levels.
2. **Scene Captioning (`captioning.py`):**
   Generates rich, syntactically descriptive summaries of the landscape, quantifying vegetation percentage, urban density, open soil, and atmospheric clarity.
3. **Visual Grounding (`grounding.py`):**
   Takes an open-ended natural phrase (e.g. *"industrial warehouse"*, *"water body"*, *"dense vegetation"*) and generates bounding coordinates $[x_1, y_1, x_2, y_2]$ with an annotated visualization.
4. **Bi-Temporal Change Detection (`change_detection.py`):**
   Compares Image $T_1$ and Image $T_2$, registers differences, computes the percentage of changed area, and exports both a binary change mask and a colorized alpha overlay.
5. **Change VQA (`change_vqa.py`):**
   Answers natural language queries about the nature of the change between two epochs (e.g., *"Has vegetation decreased?"*, *"Did urban expansion occur?"*).
6. **Optical-SAR Cross-Modal Analysis (`optical_sar.py`):**
   Fuses an optical satellite image with a Synthetic Aperture Radar (SAR) image. Leverages radar backscatter to highlight rough terrain, metallic structures, and moisture independent of cloud cover.

---

### 3.5 Visual Evidence & Confidence Calibration Engine
*Implementation:* `backend/app/services/confidence_calibration.py`, `hallucination_guard.py`

#### Calibrated Confidence Scoring:
Instead of raw uncalibrated model probabilities, SatQuery AI calculates a composite calibrated confidence score:
$$C_{\text{final}} = w_1 \cdot C_{\text{model}} + w_2 \cdot C_{\text{evidence}} + w_3 \cdot C_{\text{sensor}} - \text{Penalty}_{\text{uncertainty}}$$
- **$C_{\text{model}}$:** Base prediction probability.
- **$C_{\text{evidence}}$:** Presence and quality of visual spatial evidence (masks, boxes, feature counts).
- **$C_{\text{sensor}}$:** Metadata completeness, dynamic range, and CRS projection.
- **Output Levels:**
  - **HIGH:** $\ge 0.85$ (Strong evidence, pristine sensor metadata)
  - **MEDIUM:** $0.70 - 0.84$ (Acceptable evidence, slight sensor or spatial ambiguity)
  - **LOW:** $0.50 - 0.69$ (Inconclusive evidence, degraded resolution)
  - **UNCERTAIN:** $< 0.50$ (Fallback engaged or contradictory spatial signals)

#### Hallucination Guard:
Validates textual assertions against spatial facts. For example:
- If the text claims *"Dense forest covers the scene"*, but NDVI analysis reveals $< 5\%$ vegetation, the hallucination guard flags a semantic contradiction, lowers confidence, and appends an audit warning.

---

### 3.6 Auditable Report Generation (PDF & HTML)
*Implementation:* `backend/app/services/report_service.py`

Every completed query is permanently archived in SQLite and can be exported on demand:
- **PDF Report:** Generated via ReportLab. Includes header, ISRO/satellite metadata tables, user query, answer, confidence badges, embedded evidence images, stage-by-stage execution trace, and a legal disclaimer.
- **HTML Report:** Responsive, standalone HTML document optimized for in-browser inspection or `@media print` physical printing.
- **API Endpoints:** `GET /api/analysis/{id}/report?format=pdf` and `GET /api/analysis/{id}/report?format=html`.

---

### 3.7 Modern React Dashboard & UI Components
*Implementation:* `frontend/src/`

- **Tech Stack:** React 18, TypeScript, Tailwind CSS, Lucide Icons, Vite.
- **Design System:** Professional dark-mode aesthetic with slate backgrounds, emerald/cyan accents, and glassmorphic cards.
- **Interactive Image Comparison Slider (`ImageCompareSlider.tsx`):**
  Hardware-accelerated before/after swipe comparison slider for bi-temporal and optical-SAR pairs with touch, mouse drag, and keyboard navigation.
- **Evidence Opacity Control:** Real-time opacity slider (10% to 100%) allowing users to see through the evidence masks down to the raw satellite base layer.
- **Execution Trace Timeline:** Interactive visual timeline showing every micro-step of the query execution with millisecond counters.
- **Model Registry Modal:** Shows all registered models, task capabilities, versions, and current operational states.
- **History Drawer:** Slide-out drawer displaying past analysis sessions with retrospective report download capability.

---

## 4. Step-by-Step Life of a Query (Execution Trace)

When a user interacts with SatQuery AI, the lifecycle follows this exact sequence:

1. **Step 1: Ingestion & Upload (`POST /api/upload`)**
   - User drops one or two `.tif` files into the browser.
   - Backend writes raw data to `uploads/`.
   - `metadata.py` reads headers via Rasterio, extracts CRS, bounds, dimensions, and computes 8-bit previews into `outputs/processed/`.
   - Returns unique `upload_id` and file descriptors to the frontend.
2. **Step 2: Query Submission (`POST /api/analyze`)**
   - User types a query (e.g. *"Provide a comprehensive land-cover description"*).
   - Frontend posts `{ upload_id, query, options }`.
3. **Step 3: Agentic Task Routing**
   - `agentic_planner.py` evaluates query tokens and input modality $\rightarrow$ routes to `TaskType.CAPTIONING`.
4. **Step 4: Model Execution**
   - Registry dispatches input to `rs-scene-descriptor-cpu`.
   - Computes spectral ratios across bands, estimates vegetative coverage, and generates a structured description.
5. **Step 5: Visual Evidence Synthesis**
   - Creates colorized evidence maps in `outputs/evidence/` (e.g. vegetation density overlay or change heatmap).
6. **Step 6: Confidence Calibration & Hallucination Check**
   - Calibrates output confidence ($0.8217$, level `MEDIUM`).
   - Verifies consistency between text assertions and calculated pixel statistics.
7. **Step 7: Trace Compilation & Database Commit**
   - Records execution steps with latencies (`agent_routing`: 12ms, `model_inference`: 45ms, `evidence_generation`: 28ms).
   - Commits records to SQLite tables (`analyses`, `model_runs`, `evidence`, `execution_traces`).
8. **Step 8: UI Presentation & Report Export**
   - Frontend polls `GET /api/analysis/{id}`, animates the results, renders the evidence layer with opacity control, and provides PDF/HTML download buttons.

---

## 5. Roadmap Completion Summary (Phases 0–14)

| Phase | Name | Focus Area | Status |
|---|---|---|---|
| **0** | Project Planning | Architecture, API designs, specifications, licensing | ✅ Complete |
| **1** | Application Skeleton | FastAPI backend, React + Vite frontend, SQLite ORM | ✅ Complete |
| **2** | Geospatial Ingestion | Rasterio GeoTIFF processing, metadata, CRS, previews | ✅ Complete |
| **3** | Single-Image VQA | VQA pipeline, question-answer extraction, spectral fallback | ✅ Complete |
| **4** | Captioning | Multi-class land-cover analysis and scene descriptor | ✅ Complete |
| **5** | Visual Grounding | Phrase-to-bbox grounding, contour detection | ✅ Complete |
| **6** | Bi-Temporal Change Detection | Two-epoch comparison, Otsu differencing, change masks | ✅ Complete |
| **7** | Change VQA | Linguistic reasoning over bi-temporal modifications | ✅ Complete |
| **8** | Optical-SAR Analysis | Cross-modal optical and SAR radar backscatter fusion | ✅ Complete |
| **9** | Agentic Orchestration | Automated query routing, intent classification | ✅ Complete |
| **10**| Evidence & Confidence | Multi-factor confidence calibration, hallucination guard | ✅ Complete |
| **11**| Benchmark Evaluation | Integration with RSVQA, VRSBench, CDVQA evaluators | ✅ Complete |
| **12**| ISRO/SAC Readiness | Cartosat-2S/3 uint16 support, RISAT-1A EOS-04 SAR handling | ✅ Complete |
| **13**| Report & Audit | Multi-page ReportLab PDF & responsive HTML report export | ✅ Complete |
| **14**| Final UI/UX Polish | Swipe comparison slider, opacity control, trace timeline | ✅ Complete |

---

## 6. Codebase Organization & File Guide

```
SatQuaryAI/
├── backend/
│   ├── app/
│   │   ├── agent/                 # Agent orchestration abstractions
│   │   ├── api/
│   │   │   ├── dependencies.py    # DB session and configuration injection
│   │   │   └── routes/
│   │   │       ├── health.py      # GET /api/health
│   │   │       ├── upload.py      # POST /api/upload
│   │   │       └── analysis.py    # POST /api/analyze, GET /api/analysis/*
│   │   ├── database/
│   │   │   ├── engine.py          # SQLAlchemy engine & SQLite connection
│   │   │   ├── models.py          # ORM models (Analysis, UploadedFile, Evidence, etc.)
│   │   │   └── crud.py            # Database CRUD helper functions
│   │   ├── evaluation/            # Benchmark evaluation scripts (RSVQA, VRSBench, CDVQA)
│   │   ├── geospatial/            # Rasterio, CRS, preview generation, ISRO logic
│   │   ├── models/                # Base classes, specialist models, Model Registry
│   │   ├── services/              # Agent planner, confidence calibration, reporting
│   │   ├── utils/                 # Structured logging, YAML config, security
│   │   └── main.py                # FastAPI app creation, lifespan, CORS, static routes
│   └── tests/                     # 82 unit and API tests across all features
├── frontend/
│   ├── src/
│   │   ├── components/            # Header, UploadPanel, QueryInput, ResultPanel,
│   │   │                          # ImageCompareSlider, ModelRegistryModal, HistoryDrawer
│   │   ├── services/api.ts        # Axios/Fetch API client functions
│   │   ├── types/index.ts         # TypeScript interfaces matching backend schemas
│   │   ├── App.tsx                # Master state controller and layout
│   │   └── index.css              # Tailwind CSS directives and custom scrollbars
│   └── vite.config.ts             # Vite server configuration with /api reverse proxy
├── docs/                          # Architecture, API design, dataset specs, roadmap
├── sample_images/                 # Ready-to-use Cartosat, RISAT, and bi-temporal GeoTIFFs
├── pytest.ini                     # Pytest root configuration
├── run_backend.ps1                # PowerShell backend launcher
├── run_frontend.ps1               # PowerShell frontend launcher
└── run_all.ps1                    # 1-Click unified launcher for both servers
```

---

## 7. Verification, Testing, and Local Execution

### 7.1 Automated Tests
SatQuery AI includes an exhaustive test suite covering geospatial extraction, API endpoints, model fallbacks, confidence calibration, report generation, and ISRO satellite ingestion.

To run the complete test suite:
```powershell
.\backend\venv\Scripts\pytest.exe
```
**Result:** `82 passed` in ~5.5 seconds.

### 7.2 Running the Application
To start both servers simultaneously:
```powershell
.\run_all.ps1
```
- **Web Dashboard:** [http://localhost:5173](http://localhost:5173)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 8. Final Year Project Defense & Viva Guide (Q&A)

Here are the most common questions your project examiners and review committee will ask, along with the precise technical answers:

#### Q1: "Why did you build an agentic architecture instead of just using a single Vision-Language Model like GPT-4V or LLaVA?"
> **Answer:** *"Generic foundation models are trained on eye-level internet photos and treat images as simple RGB arrays. In remote sensing, data is multi-modal (SAR radar backscatter, multispectral NIR/SWIR bands) and requires coordinate georeferencing (CRS, UTM projections) and multi-temporal comparisons. An agentic framework allows us to intelligently parse user intent and dispatch the task to specialized domain models (e.g. bi-temporal differencing for change detection, radar despeckling for SAR, or spectral indices for vegetation), while enforcing confidence calibration and hallucination checks."*

#### Q2: "How does the system ensure the model isn't hallucinating answers?"
> **Answer:** *"SatQuery AI implements a multi-layer Hallucination Guard and Calibrated Confidence Engine. Every textual output is paired with pixel-level spatial evidence (bounding boxes or change masks). The system computes a composite confidence score based on model certainty, spatial evidence coverage, and sensor metadata quality. Furthermore, the hallucination guard cross-checks textual assertions with computed physical indices (such as NDVI for vegetation or NDWI for water); if a contradiction is detected, the confidence is penalized and an audit flag is recorded."*

#### Q3: "How does your system handle Indian space sensors (ISRO/SAC)?"
> **Answer:** *"We built dedicated ISRO readiness into Phase 12. Cartosat-2S and Cartosat-3 optical imagery feature high dynamic ranges (10/11/12-bit stored in uint16), which normally appear washed-out or pitch black in standard image viewers; our ingestion pipeline applies adaptive 2%-98% cumulative percentile stretching. For RISAT-1 and RISAT-1A (EOS-04) SAR radar imagery, the pipeline converts raw amplitude values into calibrated sigma-nought decibel backscatter ($\sigma^0_{\text{dB}}$) and handles Indian map projections including Indian UTM zones 42N-46N and Kalianpur 1975."*

#### Q4: "What happens if a high-end GPU is not available?"
> **Answer:** *"SatQuery AI is designed with a fault-tolerant dual-engine model registry. While it supports deep learning models with LoRA adaptation when a CUDA GPU is present, it features deterministic CPU specialist fallback models. These fallbacks execute algebraic spectral index calculations, Otsu thresholding, and contour approximations, guaranteeing 100% system availability, sub-second latency, and zero crashes even on standard CPU laptops."*

#### Q5: "What datasets are supported, and how can the system be evaluated quantitatively?"
> **Answer:** *"The framework integrates four benchmark remote-sensing datasets: BigEarthNet for domain vocabulary adaptation via LoRA, VRSBench for visual grounding and captioning, RSVQA for single-image visual question answering, and CDVQA for bi-temporal change reasoning. The project includes an integrated evaluation CLI (`backend/app/evaluation/runner.py`) capable of computing standard quantitative metrics including Accuracy, F1-score, Exact Match, BLEU-4, CIDEr, and Intersection-over-Union (IoU)."*
