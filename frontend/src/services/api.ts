/**
 * SatQuery AI — API service layer.
 *
 * Centralized HTTP client for all backend communication.
 */

import type {
  HealthStatus,
  AnalyzeRequest,
  AnalyzeResponse,
  AnalysisResult,
  AnalysisHistoryResponse,
  ModelInfo,
  UploadResponse,
} from '../types';


const API_BASE = '/api';

class ApiError extends Error {
  status: number;
  code: string;
  details?: Record<string, unknown>;

  constructor(status: number, code: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({
      error: 'unknown_error',
      message: `HTTP ${response.status}: ${response.statusText}`,
    }));
    throw new ApiError(
      response.status,
      errorBody.error || 'unknown_error',
      errorBody.message || 'An unexpected error occurred',
      errorBody.details,
    );
  }

  return response.json();
}

// ---- Health ----
export async function checkHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/health');
}

// ---- Upload ----
export async function uploadFiles(files: File[], inputType?: string): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  if (inputType) formData.append('input_type', inputType);

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({
      error: 'upload_error',
      message: 'Upload failed',
    }));
    throw new ApiError(response.status, errorBody.error, errorBody.message, errorBody.details);
  }

  return response.json();
}

// ---- Analyze ----
export async function submitAnalysis(data: AnalyzeRequest): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>('/analyze', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getAnalysis(analysisId: string): Promise<AnalysisResult> {
  return request<AnalysisResult>(`/analysis/${analysisId}`);
}

export async function getAnalysisTrace(analysisId: string): Promise<unknown> {
  return request(`/analysis/${analysisId}/trace`);
}

export async function getAnalysisEvidence(analysisId: string): Promise<unknown> {
  return request(`/analysis/${analysisId}/evidence`);
}

export function getReportUrl(analysisId: string, format: 'pdf' | 'html' = 'pdf'): string {
  return `${API_BASE}/analysis/${analysisId}/report?format=${format}`;
}

export async function getAnalysisHistory(limit: number = 20, offset: number = 0): Promise<AnalysisHistoryResponse> {
  return request<AnalysisHistoryResponse>(`/analysis/history?limit=${limit}&offset=${offset}`);
}

// ---- Models ----

export async function listModels(): Promise<{ models: ModelInfo[]; total: number; gpu_available: boolean }> {
  return request('/models');
}

export { ApiError };
