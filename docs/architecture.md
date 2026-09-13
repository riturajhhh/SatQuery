# SatQuery AI — System Architecture

## 1. Overview

SatQuery AI is an agentic, query-driven remote-sensing vision-language assistant. It accepts satellite imagery (optical, multispectral, SAR) along with natural-language queries and automatically routes them through validated specialist models to produce evidence-grounded answers with confidence estimation and auditable execution traces.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          USER / CLIENT                           │
│                    (Browser — React + TypeScript)                 │
└─────────────────────────────┬────────────────────────────────────┘
                              │  HTTP / WebSocket
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FASTAPI BACKEND                            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │   API Layer  │  │  Auth/Valid   │  │   File Upload Svc   │    │
│  │  (Routers)   │  │  Middleware   │  │   (sanitize, store) │    │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘    │
│         │                 │                      │               │
│         ▼                 ▼                      ▼               │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  AGENTIC CONTROLLER                      │    │
│  │                                                          │    │
│  │  1. Parse query intent                                   │    │
│  │  2. Inspect uploaded images (count, modality, metadata)  │    │
│  │  3. Validate input configuration                         │    │
│  │  4. Select tool(s) from Model Registry                   │    │
│  │  5. Execute specialist workflow                          │    │
│  │  6. Validate outputs (confidence, evidence consistency)  │    │
│  │  7. Combine evidence from multiple tools                 │    │
│  │  8. Generate final response                              │    │
│  │  9. Produce execution trace                              │    │
│  └──────────────────────────┬───────────────────────────────┘    │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │  Query/Task  │  │  Input/Modality │  │  Model/Tool     │      │
│  │  Classifier  │  │  Analyzer       │  │  Registry       │      │
│  └──────┬──────┘  └────────┬────────┘  └────────┬────────┘      │
│         │                  │                     │               │
│         ▼                  ▼                     ▼               │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              SPECIALIST MODEL WORKFLOWS                  │    │
│  │                                                          │    │
│  │  ┌─────┐ ┌─────────┐ ┌─────────┐ ┌────┐ ┌─────┐ ┌────┐│    │
│  │  │ VQA │ │Captionin│ │Grounding│ │ CD │ │CDVQA│ │O-S ││    │
│  │  └──┬──┘ └────┬────┘ └────┬────┘ └─┬──┘ └──┬──┘ └─┬──┘│    │
│  └─────┼─────────┼──────────┼────────┼───────┼──────┼────┘     │
│        └─────────┴──────────┴────────┴───────┴──────┘           │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              EVIDENCE & RESPONSE LAYER                   │    │
│  │                                                          │    │
│  │  • Evidence extraction (maps, boxes, masks)              │    │
│  │  • Result validation (confidence, consistency)           │    │
│  │  • Hallucination control                                 │    │
│  │  • Response generation (text + visual + confidence)      │    │
│  │  • Execution trace generation                            │    │
│  │  • Report generation (PDF / HTML)                        │    │
│  └──────────────────────────────────────────────────────────┘    │
│                             │                                    │
│  ┌────────────┐  ┌──────────┴───────┐  ┌──────────────────┐     │
│  │  SQLite DB  │  │  File Storage    │  │  Geospatial Svc  │     │
│  │  (analyses, │  │  (uploads,       │  │  (Rasterio,GDAL) │     │
│  │   traces)   │  │   outputs,       │  │                  │     │
│  └────────────┘  │   evidence)       │  └──────────────────┘     │
│                  └──────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

### 3.1 Frontend (React + TypeScript + Tailwind CSS)

**Purpose:** Professional dashboard for image upload, query input, result visualization, and report download.

**Key Components:**
- **Image Upload Panel** — drag-and-drop for single/paired images with format validation
- **Query Input** — natural-language text input with demo query suggestions
- **Result Display** — answer text, confidence badge, visual evidence
- **Map/Image Viewer** — interactive Leaflet-based viewer with zoom, pan, overlay, opacity control, before/after comparison
- **Execution Trace Panel** — step-by-step visualization of the agent's workflow
- **Model Information Panel** — models used, versions, parameters
- **Report Download** — PDF/HTML download button

**State Management:** React Context + useReducer for analysis state.

### 3.2 Backend (Python + FastAPI)

**Purpose:** REST API server handling uploads, query processing, model orchestration, and report generation.

**Structure:**
```
backend/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   ├── routes/
│   │   │   ├── upload.py       # POST /api/upload
│   │   │   ├── analyze.py      # POST /api/analyze
│   │   │   ├── analysis.py     # GET /api/analysis/{id}/*
│   │   │   ├── models.py       # GET /api/models
│   │   │   └── health.py       # GET /api/health
│   │   └── dependencies.py     # Shared dependencies
│   ├── agent/
│   │   ├── controller.py       # Agentic controller
│   │   ├── task_classifier.py  # Query intent classification
│   │   ├── input_analyzer.py   # Input/modality analysis
│   │   └── workflow.py         # Workflow orchestration
│   ├── models/
│   │   ├── registry.py         # Model registry
│   │   ├── base.py             # RemoteSensingModel ABC
│   │   ├── vqa.py              # VQA model adapter
│   │   ├── captioning.py       # Captioning model adapter
│   │   ├── grounding.py        # Visual grounding adapter
│   │   ├── change_detection.py # Change detection adapter
│   │   ├── change_vqa.py       # Change VQA adapter
│   │   └── optical_sar.py      # Optical-SAR fusion adapter
│   ├── preprocessing/
│   │   ├── pipeline.py         # Reusable preprocessing pipeline
│   │   ├── geotiff.py          # GeoTIFF reading (Rasterio)
│   │   ├── normalization.py    # Band normalization
│   │   ├── tiling.py           # Large image tiling
│   │   └── visualization.py    # Preview/thumbnail generation
│   ├── geospatial/
│   │   ├── metadata.py         # CRS, bounds, resolution extraction
│   │   ├── validation.py       # Co-registration, compatibility checks
│   │   └── reprojection.py     # CRS reprojection
│   ├── evaluation/
│   │   ├── run.py              # Evaluation entry point
│   │   ├── vrsbench.py         # VRSBench evaluation
│   │   ├── rsvqa.py            # RSVQA evaluation
│   │   ├── cdvqa.py            # CDVQA evaluation
│   │   ├── metrics.py          # Metric computation
│   │   └── agent_metrics.py    # Agent-specific metrics
│   ├── reports/
│   │   ├── generator.py        # PDF/HTML report generation
│   │   └── templates/          # Report templates
│   ├── database/
│   │   ├── engine.py           # SQLAlchemy engine/session
│   │   ├── models.py           # ORM models
│   │   └── crud.py             # CRUD operations
│   ├── schemas/
│   │   ├── analysis.py         # Pydantic schemas for analysis
│   │   ├── upload.py           # Upload request/response
│   │   ├── trace.py            # Execution trace schema
│   │   └── models.py           # Model info schema
│   ├── services/
│   │   ├── analysis.py         # Analysis orchestration service
│   │   ├── storage.py          # File storage service
│   │   └── evidence.py         # Evidence generation service
│   └── utils/
│       ├── config.py           # Configuration loader
│       ├── logging.py          # Structured logging
│       └── security.py         # File validation, sanitization
```

### 3.3 Agentic Controller

**Purpose:** The brain of the system — determines query intent, validates inputs, selects tools, orchestrates execution.

**Decision Flow:**
```
Query → Task Classification → Input Validation → Tool Selection → Execution → Output Validation → Response
```

**Task Classification Logic:**
1. Parse the natural-language query for intent keywords and structure
2. Determine task type: VQA / Captioning / Grounding / Change Detection / Change VQA / Optical-SAR / Unsupported
3. Verify that the input configuration (number of images, modality, temporal relationship) is compatible with the classified task
4. If incompatible, return an error with explanation

**Tool Selection:**
- Query the Model Registry for models supporting the classified task and input type
- Select the primary model (or production fallback if unavailable)
- For composite tasks (e.g., Change VQA), select multiple models in sequence

### 3.4 Model Registry

**Purpose:** Central registry of all specialist models with a common interface.

**Design Pattern:** Registry pattern — models self-register with metadata (name, version, supported inputs, supported tasks). The agent queries the registry to find compatible models.

See [model_registry.md](model_registry.md) for the full design.

### 3.5 Preprocessing Pipeline

**Purpose:** Reusable pipeline for GeoTIFF ingestion, metadata extraction, normalization, tiling, and visualization.

**Pipeline Stages:**
1. File validation (format, size, extension)
2. GeoTIFF reading via Rasterio
3. Metadata extraction (CRS, bounds, resolution, bands, dimensions)
4. Modality detection (optical vs. SAR vs. multispectral)
5. Band normalization (per-band or global)
6. Resizing (if exceeds max dimension)
7. Tiling (for large images)
8. Thumbnail/preview generation
9. Visualization generation (RGB composite, false color, etc.)

### 3.6 Geospatial Service

**Purpose:** CRS extraction, reprojection, co-registration validation, bounds checking.

### 3.7 Database (SQLite → PostgreSQL)

**ORM:** SQLAlchemy with Alembic migrations.

**Tables:**
- `analyses` — analysis ID, query, task, status, confidence, timestamps
- `uploaded_files` — file metadata, paths, modality, dimensions, CRS
- `model_runs` — model name, version, parameters, outputs
- `execution_traces` — step-by-step trace records
- `evidence` — visual evidence paths, types

### 3.8 Evaluation Module

**Purpose:** Benchmark evaluation pipelines runnable independently from the GUI.

---

## 4. Data Flow

### Single-Image VQA Flow

```
1. User uploads 1 image + types question
2. API validates upload (format, size, extension)
3. Preprocessing pipeline extracts metadata, generates preview
4. Agent classifies query → "VQA"
5. Agent checks inputs → 1 image, optical modality → compatible
6. Agent selects VQA model from registry
7. VQA model runs inference → answer + confidence
8. Evidence layer generates attention/GradCAM visualization
9. Result validation checks confidence threshold
10. Response assembled: answer + confidence + evidence + trace
11. Stored in database
12. Returned to frontend
```

### Bi-Temporal Change VQA Flow

```
1. User uploads 2 images + types question about change
2. API validates both uploads
3. Preprocessing extracts metadata for both images
4. Agent classifies query → "Change VQA"
5. Agent checks inputs → 2 images, temporal pair → compatible
6. Geospatial service validates co-registration / overlap
7. Agent selects: Change Detection model + Change VQA model
8. CD model produces change mask
9. Change VQA model answers question using both images + change mask
10. Evidence layer generates change map + highlighted regions
11. Response assembled with multi-model evidence
12. Stored and returned
```

### Optical-SAR Fusion Flow

```
1. User uploads optical + SAR images + query
2. API validates both uploads
3. Preprocessing detects modalities (optical + SAR)
4. Agent classifies query → "Optical-SAR Analysis"
5. Agent checks inputs → 1 optical + 1 SAR → compatible
6. Agent selects Optical-SAR fusion model
7. Fusion model combines features from both modalities
8. Produces complementary observations + class detections
9. Evidence layer generates fused visualization
10. Response assembled
```

---

## 5. Technology Stack Rationale

| Choice | Rationale |
|--------|-----------|
| **FastAPI** | Native async, Pydantic validation, auto-generated OpenAPI docs, excellent ML ecosystem compatibility |
| **React + TypeScript** | Component-based UI, strong typing, rich ecosystem for map/image viewers |
| **Tailwind CSS** | Rapid professional styling, consistent design system |
| **SQLite → PostgreSQL** | Zero-config for development; SQLAlchemy ORM makes migration trivial |
| **PyTorch** | Dominant ML framework, HuggingFace ecosystem, LoRA/PEFT support |
| **Rasterio + GDAL** | Industry-standard GeoTIFF processing, CRS handling, reprojection |
| **Leaflet** | Lightweight, extensible map viewer with raster overlay support |
| **Pydantic** | Schema validation for API requests/responses and configuration |
| **WeasyPrint** | HTML-to-PDF conversion for report generation |

---

## 6. Security Architecture

1. **File Upload Validation** — extension whitelist, MIME type check, magic number validation
2. **Size Limits** — configurable max upload size (default 500MB)
3. **Filename Sanitization** — strip path traversal, special characters, enforce max length
4. **Safe Storage** — uploads stored outside web root with UUID-based naming
5. **No Code Execution** — uploaded files are never executed; only read as image data
6. **Input Validation** — Pydantic schemas validate all API inputs
7. **CORS** — restricted to configured origins

---

## 7. Deployment Architecture

### Development
```
Frontend (Vite dev server :5173) → Backend (uvicorn :8000) → SQLite
```

### Production (Docker)
```
Nginx → Frontend (static) → Backend (Gunicorn + Uvicorn workers) → PostgreSQL
              ↓
        GPU Worker (model inference)
```

### Model Loading Strategy
- **Lazy loading** — models loaded on first use, not at startup
- **Model caching** — loaded models kept in memory for reuse
- **Device management** — automatic GPU/CPU selection, configurable memory limits
- **Fallback chain** — production → lightweight → demo model
