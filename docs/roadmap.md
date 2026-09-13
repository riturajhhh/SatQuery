# SatQuery AI — Development Roadmap

## Overview

This roadmap defines the sequential development phases for SatQuery AI. Each phase must be fully completed, tested, and documented before proceeding to the next.

---

## Phase Summary

| Phase | Name | Dependencies | Status |
|-------|------|-------------|--------|
| 0 | Project Planning | None | ✅ Complete |
| 1 | Application Skeleton | Phase 0 | ✅ Complete |
| 2 | Geospatial Ingestion | Phase 1 | ✅ Complete |
| 3 | Single-Image VQA | Phase 2 | ✅ Complete |
| 4 | Captioning | Phase 3 | ✅ Complete |
| 5 | Visual Grounding | Phase 3 | ✅ Complete |
| 6 | Bi-Temporal Change Detection | Phase 2 | ✅ Complete |
| 7 | Change VQA | Phase 6 | ✅ Complete |
| 8 | Optical-SAR Analysis | Phase 2 | ✅ Complete |
| 9 | Agentic Orchestration | Phases 3-8 | ✅ Complete |
| 10 | Evidence & Confidence | Phase 9 | ✅ Complete |
| 11 | Benchmark Evaluation | Phase 9 | ✅ Complete |
| 12 | ISRO/SAC Readiness | Phase 9 | ✅ Complete |
| 13 | Report & Audit | Phase 10 | ✅ Complete |
| 14 | Final UI/UX | Phase 13 | ✅ Complete |

---

## Phase 0 — Project Planning ✅

**Status:** Complete

**Deliverables:**
- [x] Architecture document (`docs/architecture.md`)
- [x] Development roadmap (`docs/roadmap.md`)
- [x] Model registry design (`docs/model_registry.md`)
- [x] API design (`docs/api_design.md`)
- [x] Dataset configuration (`docs/dataset_config.md`)
- [x] Project directory structure
- [x] Configuration files (`.env.example`, `config.yaml`)
- [x] README.md
- [x] .gitignore
- [x] LICENSE

---

## Phase 1 — Application Skeleton

**Goal:** Establish working frontend ↔ backend communication.

**Deliverables:**
- React frontend with Vite + TypeScript + Tailwind CSS
- FastAPI backend with project structure
- Basic navigation/layout
- Upload UI (drag-and-drop placeholder)
- Query input field
- `/api/health` endpoint
- SQLite database with SQLAlchemy
- Structured logging
- CORS configuration

**Acceptance Criteria:**
- Application starts successfully
- Frontend loads at `localhost:5173`
- Backend responds at `localhost:8000`
- Frontend can call `/api/health` and display the response
- Database file is created on startup

---

## Phase 2 — Geospatial Ingestion ✅

**Status:** Complete

**Deliverables:**
- [x] File upload endpoint (`POST /api/upload`)
- [x] Rasterio-based GeoTIFF processing
- [x] Metadata extraction (CRS, bounds, resolution, bands, dimensions)
- [x] Modality detection (optical / multispectral / SAR)
- [x] Preview/thumbnail generation
- [x] Compatibility validation for paired images
- [x] Upload results stored in database

**Acceptance Criteria:**
- [x] Upload a GeoTIFF → display image preview, width, height, bands, CRS, bounds, file type
- [x] Upload a non-GeoTIFF (e.g., `.exe`) → clear rejection message
- [x] Upload paired images → compatibility check runs

---

## Phase 3 — Single-Image VQA ✅

**Status:** Complete

**Deliverables:**
- [x] VQA model adapter (BLIP-2 / specialist model architecture)
- [x] LoRA/PEFT adapter for remote-sensing (BigEarthNet)
- [x] `POST /api/analyze` endpoint for VQA
- [x] Answer + confidence displayed in UI
- [x] Fallback model for CPU-only environments

**Acceptance Criteria:**
- [x] Upload satellite image + ask "What type of land cover is dominant?" → meaningful answer
- [x] Confidence score displayed
- [x] Fallback mode works on CPU

---

## Phase 4 — Captioning ✅

**Status:** Complete

**Deliverables:**
- [x] Captioning model adapter
- [x] Scene description with major objects and land-cover interpretation
- [x] Caption displayed in UI

**Acceptance Criteria:**
- [x] Upload image → request "Describe this image" → scene description returned
- [x] Description includes spatial/land-cover vocabulary

---

## Phase 5 — Visual Grounding ✅

**Status:** Complete

**Deliverables:**
- [x] Grounding model adapter (`RSGrounding_GroundingDINO` + `RSGrounding_Fallback`)
- [x] Connected-component bounding box and spatial coordinate extraction
- [x] High-resolution visual overlay generation with Pillow
- [x] Interactive grounding viewer in frontend with raw/overlay comparison toggle

**Acceptance Criteria:**
- [x] "Highlight the water body" / "Locate the vegetation" → bounding boxes drawn on image
- [x] Visual evidence overlay visible and inspectable in the UI result panel

---

## Phase 6 — Bi-Temporal Change Detection ✅

**Status:** Complete

**Deliverables:**
- [x] Change detection model adapters (`RSChangeDetection_BIT` + `RSChangeDetection_Fallback`)
- [x] Preprocessing & spatial resampling/alignment for bi-temporal image pairs
- [x] Multi-factor change mask generation (radiometric, NDVI/greenness, NDWI/water, structural gradients)
- [x] Ground area change statistics (changed percentage, pixels, $m^2$, hectares, dominant transition classification)
- [x] Change map visualization (3-panel comparative comparison banner + standalone heat overlay)
- [x] Interactive UI change viewer in ResultPanel

**Acceptance Criteria:**
- [x] Upload two temporal images → change map generated
- [x] Change statistics displayed (percentage, area in hectares, transition type)
- [x] Visualization shows changed regions with toggleable comparative view modes

---

## Phase 7 — Change VQA ✅

**Status:** Complete

**Deliverables:**
- [x] Change VQA model adapters (`RSChangeVQA_Model` + `RSChangeVQA_Fallback`)
- [x] Intent routing distinguishing between imperative change commands and interrogative change questions
- [x] Grounded natural-language reasoning citing quantitative change metrics (percentages, hectares, sector)
- [x] Integration with bi-temporal change map and evidence payload
- [x] Frontend Change VQA badge and specialized interpretation presentation

**Acceptance Criteria:**
- [x] "Did the forest area decrease?" / "What happened to the water body?" → meaningful textual answer + change evidence
- [x] Confidence score provided (calibrated HIGH)
- [x] Linked change visualization and metrics cards rendered in UI

---

## Phase 8 — Optical-SAR Analysis ✅

**Status:** Complete

**Deliverables:**
- [x] Optical-SAR fusion model adapters (`RSOpticalSAR_Model` + `RSOpticalSAR_Fallback`)
- [x] Cross-modal spatial resampling & co-registration alignment
- [x] Optical cloud detection and masking (high-albedo saturation & low color variance)
- [x] SAR microwave radar backscatter extraction (specular water, volume terrain, double-bounce structures)
- [x] Cloud penetration feature extraction under cloud cover
- [x] Fused cross-modal composite visualization (3-panel comparative comparison banner + standalone radar overlay)
- [x] Dedicated Optical-SAR Multi-Sensor Fusion panel in UI

**Acceptance Criteria:**
- [x] Upload optical + SAR pair → fused analysis result generated
- [x] Result contains synergistic observations from both modalities with verified cloud penetration
- [x] Cross-modal 3-panel and fused visualizations rendered in dashboard

---

## Phase 9 — Agentic Orchestration ✅

**Goal:** Automatic task routing — users never manually select which tool to use.

**Deliverables:**
- [x] Query intent classifier (`AgenticPlanner` with semantic rule & trigger matching)
- [x] Input configuration analyzer (single optical/SAR/multispectral, bi-temporal, optical-SAR pairs)
- [x] Automatic model/tool selection from registry
- [x] Multi-tool workflow orchestration (atomic decomposition, sequential pipeline execution, synthesis)
- [x] Execution trace generation with microsecond step profiling and UI audit timeline

**Acceptance Criteria:**
- [x] Ask any supported question → correct tool automatically selected
- [x] Execution trace shows complete workflow with detailed tool steps
- [x] Composite multi-intent queries automatically decomposed and chained across specialist models

---

## Phase 10 — Evidence & Confidence ✅

**Goal:** Every analysis produces visual evidence and calibrated confidence.

**Deliverables:**
- [x] Evidence generation for all task types (`grounding_overlay`, `change_detection_map`, `optical_sar_fusion`, `spectral_activation_summary`, `confidence_calibration`)
- [x] Confidence calibration (`ConfidenceCalibrator` with temperature scaling & raster dynamic quality assessment)
- [x] Uncertainty flags (`HIGH`, `MEDIUM`, `LOW`, `UNCERTAIN`, plus `LOW_CONTRAST_IMAGERY`, `SPECTRAL_EVIDENCE_MISMATCH`)
- [x] Evidence consistency validation (`EvidenceConsistencyValidator` checking NDVI/NDWI/Δ radiometric statistics against claims)
- [x] Hallucination control layer (`HallucinationGuard` auditing assertions, flagging discrepancies, and appending advisory warnings)

**Acceptance Criteria:**
- [x] Every supported analysis includes visual evidence and audit metadata
- [x] Low-confidence results are explicitly flagged with calibrated entropy & uncertainty reasons
- [x] "Insufficient / Inconsistent evidence" advisory warnings attached when contradictions occur

---

## Phase 11 — Benchmark Evaluation ✅

**Goal:** Reproducible evaluation on standard benchmarks.

**Deliverables:**
- [x] VRSBench evaluation pipeline (`VRSBenchEvaluator` for visual grounding and scene captioning)
- [x] RSVQA evaluation pipeline (`RSVQAEvaluator` across presence, comparison, land cover)
- [x] CDVQA evaluation pipeline (`CDVQAEvaluator` across bi-temporal change queries)
- [x] Metrics computation (Exact match, Token accuracy, BLEU-1..4, ROUGE-L, Bounding Box IoU, mAP@0.5)
- [x] Agent-specific metrics (Routing accuracy, task latency, success rate)
- [x] Results export (`runner.py` exporting `benchmark_summary.json` and `evaluation_report.md`)

**Acceptance Criteria:**
- [x] Evaluation runs independently from GUI via CLI (`python -m app.evaluation.runner`)
- [x] Metrics generated for each benchmark with stratified breakdown
- [x] Results fully reproducible and test-covered in CI suite

---

## Phase 12 — ISRO/SAC Evaluation Readiness ✅ Complete

**Goal:** Ensure the system can handle ISRO satellite data formats.

**Deliverables:**
- [x] Support for Cartosat-2S and Cartosat series optical imagery (0.65m PAN, 4-band VNIR ~1.6m GSD, uint16 10/11/12-bit dynamic range)
- [x] Support for RISAT-1 / RISAT-1A (EOS-04) / RISAT-2 SAR imagery (C-band/X-band, linear and circular/hybrid RH/RV polarimetry, calibrated $\sigma^0$ dB backscatter)
- [x] Pre-georeferenced data handling (Indian UTM Zones 42N–46N EPSG:32642–32646, Kalianpur 1975, LCC India, with fallback to unprojected pixel grids)
- [x] Co-registered pair processing (Cartosat bi-temporal and Optical-SAR pairs with sub-pixel alignment verification)
- [x] Non-assumptive ingestion engine with zero crashes on unseen or unconventional evaluation formats (graceful degradation)
- [x] Frontend ISRO badges, platform metadata display, and dedicated ISRO/SAC demo queries

**Acceptance Criteria:**
- [x] System accepts standard Indian RS satellite formats (Cartosat-2S, Cartosat-3, RISAT-1A EOS-04, Resourcesat-2A)
- [x] No crashes on unseen data formats (graceful error handling, NaN/infinite suppression, and unprojected fallback)
- [x] Verified via 9 dedicated unit tests in `test_isro_readiness.py` and 73 passing full-suite tests

---


## Phase 13 — Report & Audit ✅ Complete

**Goal:** Downloadable analysis reports with full audit trail.

**Deliverables:**
- [x] PDF report generation (ReportLab publication-ready multi-page engine)
- [x] HTML report generation (standalone, responsive, print-optimized with `@media print`)
- [x] Comprehensive audit trail compilation: query, sensor metadata, ISRO payloads, workflow, model runs, calibrated confidence, evidence, statistics, traces, limitations
- [x] Analysis history browsable in UI via `HistoryDrawer.tsx`
- [x] API routes: `GET /api/analysis/{id}/report?format=pdf|html` and `GET /api/analysis/history`

**Acceptance Criteria:**
- [x] Download PDF/HTML report for any completed analysis
- [x] Report contains all required sections (inputs, models, answer, evidence, trace, disclaimer)
- [x] Analysis history persists across sessions and enables retrospective report generation
- [x] Verified via 9 dedicated tests in `test_reporting.py` and `test_report_endpoints.py`, and 82 full-suite tests

---


## Phase 14 — Final UI/UX ✅ Complete

**Goal:** Presentation-ready professional dashboard.

**Deliverables:**
- [x] Polished dashboard with glassmorphism, shimmer loading skeletons, animated state transitions, and robust error handling
- [x] Interactive hardware-accelerated before/after swipe comparison slider (`ImageCompareSlider.tsx`) with mouse/touch drag and keyboard navigation
- [x] Evidence viewer with dynamic real-time opacity slider (10%–100%) and fullscreen zoom inspection modal
- [x] Execution trace visual timeline with stage pills, millisecond duration counters, and collapsible debug details
- [x] Model registry modal (`ModelRegistryModal.tsx`) querying `/api/models` with search and task filtering
- [x] Categorized demo queries with workflow tabs (`All`, `Single Optical`, `Bi-Temporal`, `Optical + SAR`, `ISRO/SAC`) and sensor requirement guidance chips
- [x] Responsive layout and accessibility polish (global `Escape` dismiss, ARIA labels, keyboard navigation)

**Acceptance Criteria:**
- [x] Dashboard is visually professional, presentation-ready, and adheres to dark-mode RS aesthetic
- [x] All interactive features (swipe slider, opacity control, fullscreen inspection, model registry, history drawer) work smoothly
- [x] No broken states, unhandled errors, or TypeScript/ESLint compiler warnings
- [x] All 82 full-suite backend tests passing green and frontend bundle builds cleanly in ~3s

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU unavailable | Fallback models + CPU inference + model quantization |
| Large model memory | Lazy loading + model caching + device management |
| Dataset unavailable | Demo data included + synthetic examples |
| Model performance | Pretrained models + LoRA adaptation + fallback chain |
| GeoTIFF complexity | Rasterio + GDAL for robust format handling |
