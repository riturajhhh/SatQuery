/**
 * SatQuery AI — Main Application.
 *
 * Professional dashboard layout with upload panel, query input,
 * result display, and system status.
 */

import { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import UploadPanel from './components/UploadPanel';
import QueryInput from './components/QueryInput';
import ResultPanel from './components/ResultPanel';
import HistoryDrawer from './components/HistoryDrawer';
import ModelRegistryModal from './components/ModelRegistryModal';
import { submitAnalysis, getAnalysis, getAnalysisTrace, getReportUrl } from './services/api';
import type { AnalysisResult, UploadResponse, TraceStep } from './types';

export default function App() {
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isModelsOpen, setIsModelsOpen] = useState(false);

  // Global Escape key dismiss handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsHistoryOpen(false);
        setIsModelsOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleFilesSelected = useCallback(() => {
    setResult(null);
    setTraceSteps([]);
  }, []);

  const handleUploadSuccess = useCallback((response: UploadResponse | null) => {
    setUploadResponse(response);
  }, []);

  const handleSelectHistoricalAnalysis = useCallback(async (analysisId: string) => {
    setIsAnalyzing(true);
    try {
      const analysisData = await getAnalysis(analysisId);
      setResult(analysisData);
      try {
        const traceResponse = (await getAnalysisTrace(analysisId)) as {
          analysis_id: string;
          trace: { steps: TraceStep[] };
        };
        if (traceResponse?.trace?.steps) {
          setTraceSteps(traceResponse.trace.steps);
        } else {
          setTraceSteps([]);
        }
      } catch (traceErr) {
        console.warn('Could not fetch trace:', traceErr);
        setTraceSteps([]);
      }
    } catch (err) {
      console.error('Failed to load historical analysis:', err);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleQuery = useCallback(async (query: string) => {
    if (!uploadResponse || uploadResponse.files.length === 0) {
      setResult({
        analysis_id: 'no-upload',
        status: 'failed',
        query,
        task: null,
        input_type: null,
        answer: null,
        evidence: null,
        models_used: [],
        timestamps: {},
        error: {
          code: 'NO_UPLOAD',
          message: 'Please upload at least one satellite image before asking a question.',
          suggestion: 'Drag and drop a GeoTIFF or satellite image into the input panel.',
        },
        progress: null,
      });
      return;
    }

    setIsAnalyzing(true);
    setResult(null);
    setTraceSteps([]);

    try {
      // 1. Submit query to backend
      const submission = await submitAnalysis({
        upload_id: uploadResponse.upload_id,
        query,
      });

      // 2. Fetch completed analysis result
      const analysisData = await getAnalysis(submission.analysis_id);
      setResult(analysisData);

      // 3. Fetch execution trace
      try {
        const traceResponse = (await getAnalysisTrace(submission.analysis_id)) as {
          analysis_id: string;
          trace: { steps: TraceStep[] };
        };
        if (traceResponse?.trace?.steps) {
          setTraceSteps(traceResponse.trace.steps);
        }
      } catch (traceErr) {
        console.warn('Could not fetch execution trace:', traceErr);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Analysis request failed.';
      setResult({
        analysis_id: 'error-' + Date.now(),
        status: 'failed',
        query,
        task: null,
        input_type: null,
        answer: null,
        evidence: null,
        models_used: [],
        timestamps: {},
        error: {
          code: 'ANALYSIS_ERROR',
          message: msg,
          suggestion: 'Ensure the backend server is running and your image has valid dimensions.',
        },
        progress: null,
      });
    } finally {
      setIsAnalyzing(false);
    }
  }, [uploadResponse]);

  const hasCompletedAnalysis = result && result.status === 'complete' && !result.analysis_id.startsWith('error-') && !result.analysis_id.startsWith('no-upload');

  return (
    <div className="min-h-screen flex flex-col">
      <Header
        onOpenHistory={() => setIsHistoryOpen(true)}
        onOpenModels={() => setIsModelsOpen(true)}
      />

      <main className="flex-1 max-w-[1600px] mx-auto w-full px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* ---- Left Panel: Input ---- */}
          <div className="lg:col-span-5 xl:col-span-4 space-y-6">
            <UploadPanel
              onFilesSelected={handleFilesSelected}
              onUploadSuccess={handleUploadSuccess}
            />
            <QueryInput
              onSubmit={handleQuery}
              disabled={false}
              isLoading={isAnalyzing}
            />

            {/* Audit & Report Export Card */}
            <div className="glass-card p-4 space-y-3">
              <div className="flex items-center justify-between text-xs font-semibold text-white">
                <span className="flex items-center gap-1.5">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-brand-400">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                  Official Audit Reports
                </span>
                <span className="text-[10px] text-surface-500 font-mono">Phase 13</span>
              </div>

              {hasCompletedAnalysis ? (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2">
                    <a
                      href={getReportUrl(result.analysis_id, 'pdf')}
                      download={`satquery_report_${result.analysis_id.slice(0, 8)}.pdf`}
                      className="btn-primary text-xs py-2 px-3 flex items-center justify-center gap-1.5 text-center font-medium"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                      </svg>
                      PDF Report
                    </a>
                    <a
                      href={getReportUrl(result.analysis_id, 'html')}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="btn-secondary text-xs py-2 px-3 flex items-center justify-center gap-1.5 text-center font-medium hover:border-brand-500/50"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                      </svg>
                      HTML Audit
                    </a>
                  </div>
                  <p className="text-[10px] text-surface-500 text-center">
                    Session <code>{result.analysis_id.slice(0, 8)}</code> is verified and ready for download.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <button
                    onClick={() => setIsHistoryOpen(true)}
                    className="btn-secondary w-full text-xs py-2 flex items-center justify-center gap-2 hover:border-brand-500/40"
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <polyline points="12 6 12 12 16 14" />
                    </svg>
                    Browse Previous Sessions
                  </button>
                  <p className="text-[10px] text-surface-500 text-center">
                    Run an analysis or browse previous sessions to generate tamper-evident audit reports.
                  </p>
                </div>
              )}
            </div>
          </div>


          {/* ---- Right Panel: Results ---- */}
          <div className="lg:col-span-7 xl:col-span-8">
            {!result && !isAnalyzing ? (
              /* Empty State */
              <div className="glass-card p-12 flex flex-col items-center justify-center min-h-[500px] text-center">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-500/10 to-violet-500/10 flex items-center justify-center mb-6">
                  <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="text-brand-500/60">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                    <path d="M2 12h20" />
                  </svg>
                </div>
                <h2 className="text-xl font-semibold text-surface-300 mb-2">
                  Ready for Analysis
                </h2>
                <p className="text-surface-500 text-sm max-w-md mb-8">
                  Upload satellite imagery and ask a natural-language question.
                  SatQuery AI will automatically select the appropriate specialist model
                  and return evidence-grounded results.
                </p>

                <div className="grid grid-cols-3 gap-4 w-full max-w-lg">
                  {[
                    { icon: '🛰️', label: 'VQA', desc: 'Ask questions' },
                    { icon: '🔄', label: 'Change', desc: 'Detect changes' },
                    { icon: '📡', label: 'SAR Fusion', desc: 'Multi-modal' },
                  ].map((item) => (
                    <div key={item.label} className="text-center p-4 rounded-xl bg-surface-800/20 border border-surface-800/50">
                      <span className="text-2xl">{item.icon}</span>
                      <p className="text-xs text-surface-400 font-medium mt-2">{item.label}</p>
                      <p className="text-[10px] text-surface-600 mt-0.5">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <ResultPanel
                result={result}
                traceSteps={traceSteps}
                isLoading={isAnalyzing}
              />
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-surface-800/50 py-4 mt-auto">
        <div className="max-w-[1600px] mx-auto px-6 flex items-center justify-between">
          <p className="text-[10px] text-surface-700">
            SatQuery AI v0.1.0 — Results depend on image quality. VLMs can hallucinate. Confidence ≠ absolute correctness.
          </p>
          <p className="text-[10px] text-surface-700">
            Agentic RS Vision-Language Framework
          </p>
        </div>
      </footer>


      {/* History Drawer Modal */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectAnalysis={handleSelectHistoricalAnalysis}
      />

      {/* Specialist Model Registry Modal */}
      <ModelRegistryModal
        isOpen={isModelsOpen}
        onClose={() => setIsModelsOpen(false)}
      />
    </div>
  );
}


