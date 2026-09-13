/**
 * SatQuery AI — TypeScript type definitions.
 */

// ---- Health ----
export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'error';
  version: string;
  models_loaded: number;
  gpu_available: boolean;
  database: string;
  timestamp: string;
}

// ---- ISRO / SAC Evaluation Metadata ----
export interface ISROMetadata {
  is_isro: boolean;
  platform: string;
  sensor_type: string;
  gsd_nominal_m: number;
  bit_depth_effective: number;
  polarization: string;
  band_names?: string[];
  rgb_band_indices?: number[];
  nir_composite_indices?: number[];
  indian_crs_zone?: string | null;
  raw_calibration_constant?: string | null;
}

// ---- File Upload ----
export interface FileInfo {
  file_id: string;
  original_name: string;
  stored_name: string;
  format: string | null;
  width: number | null;
  height: number | null;
  bands: number | null;
  crs: string | null;
  bounds: { left: number; bottom: number; right: number; top: number } | null;
  resolution: { x: number; y: number } | null;
  modality: string | null;
  file_size_bytes: number | null;
  preview_url: string | null;
  thumbnail_url: string | null;
  metadata: (Record<string, unknown> & { isro?: ISROMetadata | null }) | null;
}


export interface UploadResponse {
  upload_id: string;
  files: FileInfo[];
  input_type: string;
  compatibility: { valid: boolean; message: string };
  timestamp: string;
}

// ---- Analysis ----
export type AnalysisStatus = 'pending' | 'processing' | 'complete' | 'failed';
export type ConfidenceLevel = 'HIGH' | 'MEDIUM' | 'LOW' | 'UNCERTAIN';

export interface AnalyzeRequest {
  upload_id: string;
  query: string;
  options?: {
    confidence_threshold?: number;
    generate_evidence?: boolean;
    generate_report?: boolean;
  };
}

export interface AnalyzeResponse {
  analysis_id: string;
  status: AnalysisStatus;
  message: string;
  estimated_time_seconds?: number;
}

export interface AnalysisResult {
  analysis_id: string;
  status: AnalysisStatus;
  query: string;
  task: string | null;
  input_type: string | null;
  answer: {
    text: string;
    confidence: number;
    confidence_level: ConfidenceLevel;
    is_fallback: boolean;
  } | null;
  evidence: Record<string, unknown> | null;
  models_used: ModelUsedInfo[];
  timestamps: Record<string, string | null>;
  error: { code: string; message: string; suggestion?: string } | null;
  progress: { current_step: string; steps_completed: number; total_steps: number } | null;
}

export interface ModelUsedInfo {
  name: string;
  version: string | null;
  task: string;
  is_fallback: boolean;
}

// ---- Execution Trace ----
export interface TraceStep {
  step: number;
  action: string;
  status: 'success' | 'failed' | 'skipped';
  duration_ms: number | null;
  details: string | null;
}

// ---- Models ----
export interface ModelInfo {
  name: string;
  version: string | null;
  base_model: string | null;
  adapter: string | null;
  supported_tasks: string[];
  supported_inputs: string[];
  is_loaded: boolean;
  is_fallback: boolean;
  device: string | null;
}

// ---- Error ----
export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
}

// ---- Demo Queries ----
export interface DemoQuery {
  label: string;
  query: string;
  category: 'single' | 'grounding' | 'change' | 'optical_sar';
  icon: string;
}

// ---- Analysis History ----
export interface AnalysisHistoryItem {
  analysis_id: string;
  query: string;
  task: string | null;
  input_type: string | null;
  status: AnalysisStatus;
  confidence: number | null;
  confidence_level: ConfidenceLevel;
  is_fallback: boolean;
  created_at: string | null;
  completed_at: string | null;
  files_count: number;
  modalities: string[];
  thumbnail_url: string | null;
}

export interface AnalysisHistoryResponse {
  total: number;
  limit: number;
  offset: number;
  items: AnalysisHistoryItem[];
}

