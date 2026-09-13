# SatQuery AI — API Design

## 1. Overview

SatQuery AI exposes a REST API via FastAPI. All endpoints are prefixed with `/api/`.

**Base URL:** `http://localhost:8000`

**Documentation:** Available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## 2. Endpoints

### 2.1 Health Check

```
GET /api/health
```

**Purpose:** Verify backend is running and check system status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": 3,
  "gpu_available": true,
  "database": "connected",
  "timestamp": "2026-09-08T10:00:00Z"
}
```

**Status Codes:**
| Code | Description |
|------|-------------|
| 200 | System healthy |
| 503 | System degraded (partial failure) |

---

### 2.2 Upload Images

```
POST /api/upload
```

**Purpose:** Upload one or more satellite images for analysis.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | File[] | Yes | One or two image files |
| `input_type` | string | No | `single`, `bi_temporal`, `optical_sar` (auto-detected if omitted) |

**Validation:**
- Accepted formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`
- Max file size: 500MB per file
- Max 2 files per request
- Filename sanitization applied

**Response (200):**
```json
{
  "upload_id": "uuid-string",
  "files": [
    {
      "file_id": "uuid-string",
      "original_name": "image_t1.tif",
      "stored_name": "a1b2c3d4_image_t1.tif",
      "format": "GeoTIFF",
      "width": 2048,
      "height": 2048,
      "bands": 4,
      "crs": "EPSG:32643",
      "bounds": {
        "left": 72.5,
        "bottom": 23.0,
        "right": 73.0,
        "top": 23.5
      },
      "resolution": {
        "x": 10.0,
        "y": 10.0
      },
      "modality": "optical",
      "file_size_bytes": 16777216,
      "preview_url": "/api/files/uuid/preview.png",
      "thumbnail_url": "/api/files/uuid/thumbnail.png",
      "metadata": {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": null,
        "transform": [10.0, 0.0, 72.5, 0.0, -10.0, 23.5]
      }
    }
  ],
  "input_type": "single",
  "compatibility": {
    "valid": true,
    "message": "Single image upload accepted."
  },
  "timestamp": "2026-09-08T10:00:00Z"
}
```

**Error Response (400):**
```json
{
  "error": "validation_error",
  "message": "Unsupported file format: .exe. Accepted formats: .tif, .tiff, .png, .jpg, .jpeg",
  "details": {
    "file": "malware.exe",
    "reason": "unsupported_format"
  }
}
```

**Error Response (422) — Incompatible pair:**
```json
{
  "error": "compatibility_error",
  "message": "Two-image change analysis requires spatially corresponding images. The uploaded images have non-overlapping geographic bounds.",
  "details": {
    "image_a_bounds": {"left": 72.5, "bottom": 23.0, "right": 73.0, "top": 23.5},
    "image_b_bounds": {"left": 80.0, "bottom": 12.0, "right": 81.0, "top": 13.0},
    "overlap_percentage": 0.0
  }
}
```

---

### 2.3 Analyze

```
POST /api/analyze
```

**Purpose:** Submit a natural-language query for analysis.

**Request:**
```json
{
  "upload_id": "uuid-string",
  "query": "Has the built-up area increased between these two dates?",
  "options": {
    "confidence_threshold": 0.5,
    "generate_evidence": true,
    "generate_report": true
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `upload_id` | string | Yes | ID from upload response |
| `query` | string | Yes | Natural-language question |
| `options.confidence_threshold` | float | No | Min confidence (default 0.5) |
| `options.generate_evidence` | bool | No | Generate visual evidence (default true) |
| `options.generate_report` | bool | No | Generate downloadable report (default true) |

**Response (202 — Accepted, processing):**
```json
{
  "analysis_id": "uuid-string",
  "status": "processing",
  "message": "Analysis started. Poll GET /api/analysis/{id} for results.",
  "estimated_time_seconds": 30
}
```

**Long-running analyses use background processing.** The client polls `GET /api/analysis/{id}` or uses WebSocket for real-time updates.

---

### 2.4 Get Analysis Results

```
GET /api/analysis/{id}
```

**Purpose:** Retrieve completed analysis results.

**Response (200 — complete):**
```json
{
  "analysis_id": "uuid-string",
  "status": "complete",
  "query": "Has the built-up area increased?",
  "task": "change_vqa",
  "input_type": "bi_temporal",
  "answer": {
    "text": "Yes, the built-up area has increased significantly in the eastern portion of the image. New construction is visible in areas that were previously agricultural land.",
    "confidence": 0.89,
    "confidence_level": "HIGH",
    "is_fallback": false
  },
  "evidence": {
    "change_map_url": "/api/analysis/{id}/evidence/change_map.png",
    "highlighted_regions_url": "/api/analysis/{id}/evidence/highlighted.png",
    "overlay_url": "/api/analysis/{id}/evidence/overlay.png",
    "statistics": {
      "changed_area_percent": 23.4,
      "increased_builtup_percent": 18.7,
      "decreased_vegetation_percent": 15.2
    }
  },
  "models_used": [
    {
      "name": "bit-cd",
      "version": "1.0.0",
      "task": "change_detection",
      "is_fallback": false
    },
    {
      "name": "cdvqa-model",
      "version": "1.0.0",
      "task": "change_vqa",
      "is_fallback": false
    }
  ],
  "timestamps": {
    "submitted": "2026-09-08T10:00:00Z",
    "started": "2026-09-08T10:00:01Z",
    "completed": "2026-09-08T10:00:28Z"
  }
}
```

**Response (200 — processing):**
```json
{
  "analysis_id": "uuid-string",
  "status": "processing",
  "progress": {
    "current_step": "change_detection",
    "steps_completed": 3,
    "total_steps": 6
  }
}
```

**Response (200 — failed):**
```json
{
  "analysis_id": "uuid-string",
  "status": "failed",
  "error": {
    "code": "model_error",
    "message": "Change detection model failed due to insufficient GPU memory.",
    "suggestion": "Try reducing image dimensions or use CPU fallback mode."
  }
}
```

---

### 2.5 Get Execution Trace

```
GET /api/analysis/{id}/trace
```

**Purpose:** Retrieve the machine-readable execution trace.

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "trace": {
    "query": "Has the built-up area increased?",
    "task": "change_vqa",
    "input_type": "bi_temporal",
    "steps": [
      {
        "step": 1,
        "action": "input_validation",
        "status": "success",
        "duration_ms": 45,
        "details": "2 images validated: bi-temporal pair, optical modality"
      },
      {
        "step": 2,
        "action": "query_classification",
        "status": "success",
        "duration_ms": 12,
        "details": "Classified as change_vqa task"
      },
      {
        "step": 3,
        "action": "model_selection",
        "status": "success",
        "duration_ms": 5,
        "details": "Selected: bit-cd (change_detection), cdvqa-model (change_vqa)"
      },
      {
        "step": 4,
        "action": "change_detection",
        "status": "success",
        "duration_ms": 8500,
        "details": "Change mask generated, 23.4% area changed"
      },
      {
        "step": 5,
        "action": "change_vqa",
        "status": "success",
        "duration_ms": 12000,
        "details": "Answer generated with confidence 0.89"
      },
      {
        "step": 6,
        "action": "evidence_generation",
        "status": "success",
        "duration_ms": 2000,
        "details": "Change map and highlighted regions generated"
      },
      {
        "step": 7,
        "action": "response_validation",
        "status": "success",
        "duration_ms": 50,
        "details": "Answer consistent with evidence, confidence HIGH"
      }
    ],
    "total_duration_ms": 22612,
    "models": ["bit-cd", "cdvqa-model"],
    "parameters": {
      "confidence_threshold": 0.5,
      "cd_threshold": 0.5
    },
    "status": "success",
    "confidence": 0.89
  }
}
```

---

### 2.6 Get Evidence

```
GET /api/analysis/{id}/evidence
```

**Purpose:** List all visual evidence generated for an analysis.

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "evidence": [
    {
      "type": "change_map",
      "url": "/api/analysis/{id}/evidence/change_map.png",
      "description": "Binary change mask showing changed regions"
    },
    {
      "type": "highlighted_regions",
      "url": "/api/analysis/{id}/evidence/highlighted.png",
      "description": "Original image with changed regions highlighted"
    },
    {
      "type": "overlay",
      "url": "/api/analysis/{id}/evidence/overlay.png",
      "description": "Change overlay on temporal image pair"
    }
  ]
}
```

```
GET /api/analysis/{id}/evidence/{filename}
```

**Purpose:** Download a specific evidence file.

**Response:** Binary image file.

---

### 2.7 Download Report

```
GET /api/analysis/{id}/report
```

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | string | `pdf` | `pdf` or `html` |

**Response:** Binary PDF or HTML file.

---

### 2.8 List Models

```
GET /api/models
```

**Purpose:** List all registered models with metadata.

**Response:**
```json
{
  "models": [
    {
      "name": "blip2-rs-vqa",
      "version": "1.0.0",
      "base_model": "Salesforce/blip2-opt-2.7b",
      "adapter": "lora",
      "supported_tasks": ["vqa"],
      "supported_inputs": ["single_optical", "single_multispectral"],
      "is_loaded": true,
      "is_fallback": false,
      "device": "cuda:0"
    },
    {
      "name": "bit-cd",
      "version": "1.0.0",
      "base_model": "BIT",
      "supported_tasks": ["change_detection"],
      "supported_inputs": ["bi_temporal"],
      "is_loaded": false,
      "is_fallback": false,
      "device": "auto"
    }
  ],
  "total": 8,
  "gpu_available": true
}
```

---

## 3. Error Handling

All errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human-readable error description",
  "details": {}
}
```

### Standard Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `validation_error` | 400 | Invalid input (format, size, etc.) |
| `compatibility_error` | 422 | Incompatible image pair |
| `not_found` | 404 | Analysis/resource not found |
| `model_error` | 500 | Model inference failure |
| `unsupported_task` | 400 | Query task not supported |
| `timeout` | 504 | Analysis exceeded time limit |
| `server_error` | 500 | Unexpected server error |

---

## 4. File Serving

Static files (previews, evidence, reports) are served via:

```
GET /api/files/{file_id}/{filename}
```

Files are stored with UUID-based paths to prevent path traversal.

---

## 5. Authentication

For the initial prototype, no authentication is required. The API is designed to support future addition of:

- API key authentication
- JWT tokens
- OAuth2

via FastAPI's dependency injection system.

---

## 6. CORS Configuration

Allowed origins are configured via the `ALLOWED_ORIGINS` environment variable:

```
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

---

## 7. Rate Limiting

Not implemented in Phase 0. Designed to be added via FastAPI middleware in production:

```python
# Future: slowapi or custom middleware
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    ...
```
