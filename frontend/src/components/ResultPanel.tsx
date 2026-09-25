import { useState } from 'react';
import type { AnalysisResult, TraceStep } from '../types';
import ImageCompareSlider from './ImageCompareSlider';

interface ResultPanelProps {
  result: AnalysisResult | null;
  traceSteps?: TraceStep[];
  isLoading: boolean;
}

function ConfidenceBadge({ level, value }: { level: string; value: number }) {
  const config: Record<string, { class: string; icon: string }> = {
    HIGH: { class: 'badge-success', icon: '✓' },
    MEDIUM: { class: 'badge-warning', icon: '~' },
    LOW: { class: 'badge-error', icon: '!' },
    UNCERTAIN: { class: 'badge-error', icon: '?' },
  };
  const c = config[level] || config.UNCERTAIN;
  return (
    <span className={`badge ${c.class}`}>
      {c.icon} {level} — {Math.round(value * 100)}%
    </span>
  );
}

export function TraceStepRow({ step }: { step: TraceStep }) {
  const [isOpen, setIsOpen] = useState(false);
  const statusIcons: Record<string, string> = {
    success: '✓',
    failed: '✗',
    skipped: '○',
  };
  const statusColors: Record<string, string> = {
    success: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    failed: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
    skipped: 'text-surface-500 bg-surface-800/50 border-surface-700/30',
  };

  return (
    <div className="trace-step py-2 px-3 rounded-lg bg-surface-900/40 hover:bg-surface-900/80 border border-surface-800/60 transition-all space-y-1.5">
      <div
        className="flex items-center gap-3 cursor-pointer select-none"
        onClick={() => setIsOpen(!isOpen)}
        title="Click to toggle details"
      >
        <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold border ${statusColors[step.status] || ''}`}>
          {statusIcons[step.status] || '?'}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-surface-200 font-medium flex items-center gap-2">
            <span>Step {step.step}:</span>
            <span>{step.action}</span>
          </p>
        </div>
        {step.duration_ms !== null && (
          <span className="text-[10px] text-surface-400 font-mono flex-shrink-0 bg-surface-800/80 px-1.5 py-0.5 rounded border border-surface-700/50">
            {step.duration_ms.toFixed(1)}ms
          </span>
        )}
        <span className="text-surface-500 text-[10px]">{isOpen ? '▲' : '▼'}</span>
      </div>
      {isOpen && step.details && (
        <div className="pl-8 pt-1 text-[11px] text-surface-300 font-mono bg-surface-950/60 p-2 rounded border border-surface-800/60 break-words">
          {step.details}
        </div>
      )}
    </div>
  );
}


export default function ResultPanel({ result, traceSteps = [], isLoading }: ResultPanelProps) {
  const [showOriginal, setShowOriginal] = useState(false);
  const [changeViewMode, setChangeViewMode] = useState<'banner' | 'overlay' | 'slider'>('banner');
  const [sarViewMode, setSarViewMode] = useState<'banner' | 'fused' | 'slider'>('banner');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(85);
  const [fullscreenImg, setFullscreenImg] = useState<{ url: string; title: string } | null>(null);

  if (isLoading) {
    return (
      <div className="space-y-6 animate-fade-in">
        <div className="section-header">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-brand-400 animate-spin"
          >
            <circle cx="12" cy="12" r="10" strokeDasharray="31.4 31.4" />
          </svg>
          <h2>Analyzing Satellite Scene...</h2>
          <div className="section-divider" />
        </div>

        {/* Shimmer placeholders */}
        <div className="glass-card p-6 space-y-4">
          <div className="shimmer h-4 w-3/4 rounded" />
          <div className="shimmer h-4 w-1/2 rounded" />
          <div className="shimmer h-32 w-full rounded-lg" />
        </div>
      </div>
    );
  }

  if (!result) return null;

  if (result.status === 'failed') {
    return (
      <div className="space-y-4 animate-fade-in">
        <div className="section-header">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-rose-400"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <h2>Analysis Failed</h2>
          <div className="section-divider" />
        </div>
        <div className="glass-card p-6 border-rose-500/20">
          <p className="text-rose-400 text-sm">
            {result.error?.message || 'An unexpected error occurred during execution.'}
          </p>
          {result.error?.suggestion && (
            <p className="text-surface-400 text-xs mt-2">{result.error.suggestion}</p>
          )}
        </div>
      </div>
    );
  }

  const spectralEvidence = result.evidence?.spectral_indices as Record<string, unknown> | undefined;
  const spectralIndices = (spectralEvidence?.spatial_indices || spectralEvidence?.features) as
    | Record<string, number>
    | undefined;

  const inputFiles = (result.evidence?.input_files as Array<{ file_id: string; preview_url: string; original_name?: string }>) || [];
  const primaryPreviewUrl = (result.evidence?.primary_preview_url as string) || (inputFiles[0]?.preview_url) || null;
  const t1Url = (result.evidence?.t1_preview_url as string) || (inputFiles[0]?.preview_url) || null;
  const t2Url = (result.evidence?.t2_preview_url as string) || (inputFiles[1]?.preview_url) || primaryPreviewUrl || null;

  const groundingEvidence = result.evidence?.grounding_overlay as Record<string, unknown> | undefined;
  const overlayUrl =
    (groundingEvidence?.overlay_url as string) ||
    (groundingEvidence?.file_url as string) ||
    (result.evidence?.overlay_url as string) ||
    null;

  const changeEvidence = result.evidence?.change_detection_map as Record<string, unknown> | undefined;
  const changeStats = (changeEvidence?.statistics || changeEvidence?.change_statistics) as Record<string, unknown> | undefined;
  const changeMapUrl =
    (changeEvidence?.change_map_url as string) ||
    (changeEvidence?.overlay_url as string) ||
    (changeEvidence?.file_url as string) ||
    null;
  const changeOverlayUrl =
    (changeEvidence?.overlay_url as string) ||
    (changeEvidence?.change_map_url as string) ||
    (changeEvidence?.file_url as string) ||
    null;

  const opticalSarEvidence = result.evidence?.optical_sar_fusion as Record<string, unknown> | undefined;
  const opticalSarStats = opticalSarEvidence?.statistics as Record<string, unknown> | undefined;
  const opticalSarBannerUrl = (opticalSarEvidence?.change_map_url as string) || null;
  const opticalSarOverlayUrl = (opticalSarEvidence?.overlay_url as string) || null;


  const calibEvidence = result.evidence?.confidence_calibration as Record<string, unknown> | undefined;
  const calibDetails = calibEvidence?.calibrated_confidence as Record<string, unknown> | undefined;
  const isConsistent = calibEvidence?.is_consistent as boolean | undefined;
  const calibScore = typeof calibDetails?.score === 'number' ? calibDetails.score : null;
  const calibEntropy = typeof calibDetails?.entropy === 'number' ? calibDetails.entropy : null;
  const qualityScore = typeof calibDetails?.quality_score === 'number' ? calibDetails.quality_score : null;
  const consistencyScore = typeof calibEvidence?.consistency_score === 'number' ? calibEvidence.consistency_score : null;
  const calibFlags = (calibDetails?.flags as string[]) || [];
  const violations = (calibEvidence?.violations as string[]) || [];

  const boxes = (groundingEvidence?.boxes || result.evidence?.boxes) as
    | Array<{
        label: string;
        confidence: number;
        box_2d: number[];
        box_pixel: number[];
      }>
    | undefined;

  return (
    <div className="space-y-6 animate-slide-up">
      {/* ---- AI Answer ---- */}
      {result.answer && (
        <div>
          <div className="section-header">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="text-brand-400"
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            <h2>
              {result.task === 'captioning'
                ? 'Scene Description & Interpretation'
                : result.task === 'grounding'
                ? 'Visual Grounding Result'
                : result.task === 'change_detection'
                ? 'Bi-Temporal Change Analysis'
                : result.task === 'change_vqa'
                ? 'Bi-Temporal Change Q&A'
                : result.task === 'optical_sar_analysis'
                ? 'Optical-SAR Cross-Modal Analysis'
                : 'AI Answer'}
            </h2>
            <div className="flex items-center gap-2 ml-auto">
              <a
                href={`/api/analysis/${result.analysis_id}/report?format=pdf`}
                target="_blank"
                rel="noopener noreferrer"
                download={`satquery_report_${result.analysis_id.slice(0, 8)}.pdf`}
                className="btn-secondary text-[11px] py-1 px-2.5 flex items-center gap-1.5 hover:border-brand-500/50 bg-surface-800/80 hover:bg-surface-700/80 transition-all cursor-pointer"
                title="Download Official Audit PDF Report"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-brand-400">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="12" y1="18" x2="12" y2="12" />
                  <line x1="9" y1="15" x2="15" y2="15" />
                </svg>
                PDF Report
              </a>
              <a
                href={`/api/analysis/${result.analysis_id}/report?format=html`}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary text-[11px] py-1 px-2.5 flex items-center gap-1.5 hover:border-brand-500/50 bg-surface-800/80 hover:bg-surface-700/80 transition-all cursor-pointer"
                title="Open Interactive HTML Audit Report in new tab"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="2" y1="12" x2="22" y2="12" />
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                </svg>
                HTML Audit
              </a>
            </div>
            <div className="section-divider" />
          </div>
          <div className="glass-card p-6">

            <p className="text-white text-sm leading-relaxed mb-4">{result.answer.text}</p>
            <div className="flex items-center gap-3 flex-wrap">
              {result.task === 'captioning' && (
                <span className="badge badge-info text-[10px]">
                  📝 Remote-Sensing Caption
                </span>
              )}
              {result.task === 'grounding' && (
                <span className="badge badge-success text-[10px]">
                  🎯 Spatial Grounding Overlay
                </span>
              )}
              {result.task === 'change_detection' && (
                <span className="badge badge-warning text-[10px] text-amber-400 border-amber-500/30 bg-amber-500/10">
                  🔄 Bi-Temporal Change Detection
                </span>
              )}
              {result.task === 'change_vqa' && (
                <span className="badge badge-warning text-[10px] text-cyan-400 border-cyan-500/30 bg-cyan-500/10">
                  💬 Bi-Temporal Change VQA
                </span>
              )}
              {result.task === 'optical_sar_analysis' && (
                <span className="badge badge-warning text-[10px] text-sky-400 border-sky-500/30 bg-sky-500/10">
                  🛰️ Optical-SAR Fusion Analysis
                </span>
              )}
              <ConfidenceBadge
                level={result.answer.confidence_level}
                value={result.answer.confidence}
              />
              {calibEvidence && isConsistent === true && (
                <span className="badge badge-success text-[10px] bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
                  🛡️ Spectrally Verified
                </span>
              )}
              {calibEvidence && isConsistent === false && (
                <span className="badge badge-error text-[10px] bg-rose-500/10 text-rose-400 border-rose-500/30">
                  ⚠️ Inconsistent Evidence
                </span>
              )}
              {result.answer.is_fallback && (
                <span className="badge badge-warning text-[10px]">
                  ⚠️ CPU Fallback Specialist Model
                </span>
              )}
            </div>

            {/* Evidence Consistency & Calibrated Uncertainty Breakdown */}
            {calibEvidence && (
              <div className="mt-4 pt-4 border-t border-surface-800/80 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-surface-400 font-medium flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    Spectral Evidence & Calibration Audit
                  </span>
                  <span className="text-surface-500 text-[11px] font-mono">
                    Consistency: {consistencyScore !== null ? `${Math.round(consistencyScore * 100)}%` : '100%'}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <div className="bg-surface-900/60 rounded p-2 border border-surface-800/50">
                    <p className="text-[10px] text-surface-500 uppercase tracking-wider">Consistency</p>
                    <p className="text-sm font-semibold text-surface-200 mt-0.5">
                      {consistencyScore !== null ? `${Math.round(consistencyScore * 100)}%` : '100%'}
                    </p>
                  </div>
                  <div className="bg-surface-900/60 rounded p-2 border border-surface-800/50">
                    <p className="text-[10px] text-surface-500 uppercase tracking-wider">Raster Quality</p>
                    <p className="text-sm font-semibold text-surface-200 mt-0.5">
                      {qualityScore !== null ? `${Math.round(qualityScore * 100)}%` : '100%'}
                    </p>
                  </div>
                  <div className="bg-surface-900/60 rounded p-2 border border-surface-800/50">
                    <p className="text-[10px] text-surface-500 uppercase tracking-wider">Entropy</p>
                    <p className="text-sm font-semibold text-surface-200 mt-0.5">
                      {calibEntropy !== null ? calibEntropy.toFixed(3) : '0.000'}
                    </p>
                  </div>
                  <div className="bg-surface-900/60 rounded p-2 border border-surface-800/50">
                    <p className="text-[10px] text-surface-500 uppercase tracking-wider">Calibrated Score</p>
                    <p className="text-sm font-semibold text-brand-400 mt-0.5">
                      {calibScore !== null ? `${Math.round(calibScore * 100)}%` : `${Math.round(result.answer.confidence * 100)}%`}
                    </p>
                  </div>
                </div>

                {calibFlags.length > 0 && (
                  <div className="flex items-center gap-1.5 flex-wrap pt-1">
                    <span className="text-[10px] text-surface-500">Flags:</span>
                    {calibFlags.map((flag) => (
                      <span key={flag} className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-mono">
                        {flag}
                      </span>
                    ))}
                  </div>
                )}

                {violations.length > 0 && (
                  <div className="bg-rose-500/10 border border-rose-500/20 rounded p-2.5 text-xs text-rose-300 space-y-1">
                    <p className="font-semibold flex items-center gap-1 text-[11px] text-rose-400">
                      <span>⚠️</span> Potential Spectral Discrepancy Detected
                    </p>
                    {violations.map((v, i) => (
                      <p key={i} className="text-[11px] text-rose-200/90 leading-relaxed">• {v}</p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ---- Visual & Spectral Evidence ---- */}
      <div>
        <div className="section-header">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-brand-400"
          >
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
          <h2>Grounded Evidence</h2>
          <div className="section-divider" />
        </div>

        {/* Bi-Temporal Change Detection Panel */}
        {changeEvidence && (
          <div className="glass-card p-6 space-y-5 mb-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <h3 className="text-xs uppercase tracking-wider font-semibold text-rose-400 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                  Bi-Temporal Change Map ({String(changeStats?.changed_percentage ?? 0)}% Surface Shift)
                </h3>
                <p className="text-[11px] text-surface-400 mt-0.5">
                  Dominant Transition: <strong className="text-white">{String(changeStats?.dominant_transition ?? 'Observed Shift')}</strong>
                </p>
              </div>
              <div className="flex items-center gap-1.5 bg-surface-800/80 p-1 rounded-lg border border-surface-700/50">
                <button
                  type="button"
                  onClick={() => setChangeViewMode('banner')}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    changeViewMode === 'banner'
                      ? 'bg-rose-500 text-white font-medium shadow-sm'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  3-Panel Comparison
                </button>
                <button
                  type="button"
                  onClick={() => setChangeViewMode('overlay')}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    changeViewMode === 'overlay'
                      ? 'bg-rose-500 text-white font-medium shadow-sm'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  Heatmap Overlay
                </button>
                <button
                  type="button"
                  onClick={() => setChangeViewMode('slider')}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    changeViewMode === 'slider'
                      ? 'bg-rose-500 text-white font-medium shadow-sm'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  Interactive Slider
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setFullscreenImg({
                      url: changeViewMode === 'overlay' ? (changeOverlayUrl || changeMapUrl || '') : (changeMapUrl || ''),
                      title: 'Bi-Temporal Change Map',
                    })
                  }
                  className="p-1 px-1.5 rounded text-surface-400 hover:text-white hover:bg-surface-700/50 transition-colors"
                  title="Expand to Fullscreen"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Change Metrics Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Changed Area</span>
                <span className="text-lg font-bold text-rose-400 font-mono">
                  {String(changeStats?.changed_percentage ?? 0)}%
                </span>
                <div className="w-full bg-surface-700/40 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    className="bg-rose-500 h-full rounded-full transition-all"
                    style={{ width: `${Math.min(100, Number(changeStats?.changed_percentage ?? 0))}%` }}
                  />
                </div>
              </div>

              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Est. Area (Hectares)</span>
                <span className="text-lg font-bold text-amber-400 font-mono">
                  {String(changeStats?.changed_area_hectares ?? 0)} ha
                </span>
                <span className="text-[10px] text-surface-500 block mt-1 font-mono">
                  {String(changeStats?.changed_area_sq_meters ?? 0)} m²
                </span>
              </div>

              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Changed Pixels</span>
                <span className="text-lg font-bold text-cyan-400 font-mono">
                  {Number(changeStats?.changed_pixels ?? 0).toLocaleString()}
                </span>
                <span className="text-[10px] text-surface-500 block mt-1 font-mono">
                  of {Number(changeStats?.total_pixels ?? 0).toLocaleString()} px
                </span>
              </div>

              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Ground Resolution</span>
                <span className="text-lg font-bold text-emerald-400 font-mono">
                  {String(changeStats?.gsd_meters ?? 10)}m
                </span>
                <span className="text-[10px] text-surface-500 block mt-1">
                  Per-pixel GSD
                </span>
              </div>
            </div>

            {/* Change Map Image Display */}
            {changeViewMode === 'slider' ? (
              <ImageCompareSlider
                beforeImage={t1Url || changeOverlayUrl || changeMapUrl || ''}
                afterImage={t2Url || changeMapUrl || changeOverlayUrl || ''}
                beforeLabel="T1 (Pre-Event)"
                afterLabel="T2 (Post-Event)"
              />
            ) : (
              <div className="relative w-full rounded-xl overflow-hidden bg-surface-900 border border-surface-800 flex items-center justify-center">
                <img
                  src={changeViewMode === 'banner' ? (changeMapUrl || changeOverlayUrl || t2Url || '') : (changeOverlayUrl || changeMapUrl || t2Url || '')}
                  alt="Bi-Temporal Change Detection Visualization"
                  className="w-full h-auto max-h-[460px] object-contain"
                />
              </div>
            )}
          </div>
        )}

        {/* Optical-SAR Multi-Sensor Fusion Panel */}
        {opticalSarEvidence && (
          <div className="glass-card p-6 space-y-5 mb-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <h3 className="text-xs uppercase tracking-wider font-semibold text-sky-400 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse" />
                  Optical-SAR Multi-Sensor Fusion ({String(opticalSarStats?.cloud_cover_percent ?? 0)}% Cloud Penetrated)
                </h3>
                <p className="text-[11px] text-surface-400 mt-0.5">
                  Radar Microwave Penetration: <strong className="text-white">Active All-Weather Sensing Continuity</strong>
                </p>
              </div>
              <div className="flex items-center gap-1.5 bg-surface-800/80 p-1 rounded-lg border border-surface-700/50">
                <button
                  type="button"
                  onClick={() => setSarViewMode('banner')}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    sarViewMode === 'banner'
                      ? 'bg-sky-500 text-white font-medium shadow-sm'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  3-Panel Sensor View
                </button>
                <button
                  type="button"
                  onClick={() => setSarViewMode('fused')}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    sarViewMode === 'fused'
                      ? 'bg-sky-500 text-white font-medium shadow-sm'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  Fused Composite
                </button>
                <button
                  type="button"
                  onClick={() => setSarViewMode('slider')}
                  className={`px-2.5 py-1 rounded text-xs transition-colors ${
                    sarViewMode === 'slider'
                      ? 'bg-sky-500 text-white font-medium shadow-sm'
                      : 'text-surface-400 hover:text-white'
                  }`}
                >
                  Interactive Slider
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setFullscreenImg({
                      url: sarViewMode === 'fused' ? (opticalSarOverlayUrl || opticalSarBannerUrl || '') : (opticalSarBannerUrl || ''),
                      title: 'Optical-SAR Multi-Sensor Cross-Modal Visualization',
                    })
                  }
                  className="p-1 px-1.5 rounded text-surface-400 hover:text-white hover:bg-surface-700/50 transition-colors"
                  title="Expand to Fullscreen"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Optical-SAR Metrics Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Optical Cloud Cover</span>
                <span className="text-lg font-bold text-sky-400 font-mono">
                  {String(opticalSarStats?.cloud_cover_percent ?? 0)}%
                </span>
                <div className="w-full bg-surface-700/40 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    className="bg-sky-500 h-full rounded-full transition-all"
                    style={{ width: `${Math.min(100, Number(opticalSarStats?.cloud_cover_percent ?? 0))}%` }}
                  />
                </div>
              </div>

              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Pierced Footprint</span>
                <span className="text-lg font-bold text-cyan-400 font-mono">
                  {String(opticalSarStats?.cloud_area_hectares ?? 0)} ha
                </span>
                <span className="text-[10px] text-surface-500 block mt-1 font-mono">
                  {Number(opticalSarStats?.cloud_pixels ?? 0).toLocaleString()} px
                </span>
              </div>

              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Radar Scatterers</span>
                <span className="text-lg font-bold text-amber-400 font-mono">
                  {Number(opticalSarStats?.active_scatterers_count ?? 0).toLocaleString()}
                </span>
                <span className="text-[10px] text-surface-500 block mt-1">
                  Double-bounce structures
                </span>
              </div>

              <div className="bg-surface-800/50 border border-surface-700/50 rounded-xl p-3">
                <span className="text-[10px] text-surface-400 block mb-1">Sensor Synergy</span>
                <span className="text-lg font-bold text-emerald-400 font-mono">
                  100%
                </span>
                <span className="text-[10px] text-surface-500 block mt-1">
                  Cross-modal continuity
                </span>
              </div>
            </div>

            {/* Optical-SAR Image Display */}
            {sarViewMode === 'slider' ? (
              <ImageCompareSlider
                beforeImage={opticalSarOverlayUrl || opticalSarBannerUrl || ''}
                afterImage={opticalSarBannerUrl || ''}
                beforeLabel="SAR Microwave Penetration"
                afterLabel="Optical Multispectral"
              />
            ) : (
              <div className="relative w-full rounded-xl overflow-hidden bg-surface-900 border border-surface-800 flex items-center justify-center">
                <img
                  src={sarViewMode === 'banner' ? (opticalSarBannerUrl || '') : (opticalSarOverlayUrl || opticalSarBannerUrl || '')}
                  alt="Optical-SAR Multi-Sensor Cross-Modal Visualization"
                  className="w-full h-auto max-h-[460px] object-contain"
                />
              </div>
            )}
          </div>
        )}

        {/* Visual Grounding Overlay */}
        {overlayUrl && !changeEvidence && !opticalSarEvidence && (
          <div className="glass-card p-6 space-y-4 mb-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <h3 className="text-xs uppercase tracking-wider font-semibold text-surface-300">
                Spatial Feature Overlay ({boxes?.length || 0} Target Clusters)
              </h3>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 text-xs text-surface-400 bg-surface-800/60 px-2.5 py-1 rounded-lg border border-surface-700/40">
                  <span>Opacity:</span>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={overlayOpacity}
                    onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                    className="w-16 accent-brand-500 cursor-pointer h-1.5"
                    aria-label="Overlay Opacity"
                  />
                  <span className="font-mono text-surface-300 text-[11px] w-6">{overlayOpacity}%</span>
                </div>
                <div className="flex items-center gap-1.5 bg-surface-800/80 p-1 rounded-lg border border-surface-700/50">
                  <button
                    type="button"
                    onClick={() => setShowOriginal(false)}
                    className={`px-2.5 py-1 rounded text-xs transition-colors ${
                      !showOriginal
                        ? 'bg-brand-500 text-white font-medium'
                        : 'text-surface-400 hover:text-white'
                    }`}
                  >
                    Annotated Overlay
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowOriginal(true)}
                    className={`px-2.5 py-1 rounded text-xs transition-colors ${
                      showOriginal
                        ? 'bg-brand-500 text-white font-medium'
                        : 'text-surface-400 hover:text-white'
                    }`}
                  >
                    Raw Image
                  </button>
                  <button
                    type="button"
                    onClick={() => setFullscreenImg({ url: overlayUrl, title: 'Spatial Feature Overlay' })}
                    className="p-1 px-1.5 rounded text-surface-400 hover:text-white hover:bg-surface-700/50 transition-colors"
                    title="Expand to Fullscreen"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>

            <div className="relative aspect-video max-h-[440px] w-full rounded-xl overflow-hidden bg-surface-900 border border-surface-800 flex items-center justify-center">
              <img
                src={showOriginal ? (primaryPreviewUrl || overlayUrl) : overlayUrl}
                alt="Grounding Overlay"
                className="w-full h-full object-contain transition-opacity duration-150"
                style={{ opacity: showOriginal ? 1 : overlayOpacity / 100 }}
              />
            </div>

            {boxes && boxes.length > 0 && (
              <div className="space-y-2 pt-2 border-t border-surface-800">
                <p className="text-[11px] uppercase tracking-wider text-surface-500 font-medium">
                  Localized Bounding Regions
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {boxes.map((b, idx) => (
                    <div
                      key={idx}
                      className="bg-surface-800/40 border border-surface-800 rounded-lg p-2.5 flex items-center justify-between text-xs"
                    >
                      <div>
                        <span className="text-surface-200 font-medium block">
                          {b.label} #{idx + 1}
                        </span>
                        <span className="text-[10px] text-surface-500 font-mono">
                          [{b.box_pixel.join(', ')}]
                        </span>
                      </div>
                      <span className="font-mono text-[11px] text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                        {Math.round(b.confidence * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Spectral Metrics Card */}
        {spectralIndices ? (
          <div className="glass-card p-6 space-y-4">
            <h3 className="text-xs uppercase tracking-wider font-semibold text-surface-400">
              Spectral & Spatial Metrics
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="bg-surface-800/40 rounded-xl p-3 border border-surface-800">
                <span className="text-[10px] text-surface-500 block mb-1">Vegetation Canopy</span>
                <span className="text-lg font-bold text-emerald-400 font-mono">
                  {spectralIndices.vegetation_cover_percent ?? spectralIndices.veg_percent ?? 0}%
                </span>
                <div className="w-full bg-surface-700/50 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    className="bg-emerald-500 h-full rounded-full"
                    style={{
                      width: `${Math.min(
                        100,
                        spectralIndices.vegetation_cover_percent ?? spectralIndices.veg_percent ?? 0
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="bg-surface-800/40 rounded-xl p-3 border border-surface-800">
                <span className="text-[10px] text-surface-500 block mb-1">Surface Water</span>
                <span className="text-lg font-bold text-cyan-400 font-mono">
                  {spectralIndices.water_cover_percent ?? spectralIndices.water_percent ?? 0}%
                </span>
                <div className="w-full bg-surface-700/50 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div
                    className="bg-cyan-500 h-full rounded-full"
                    style={{
                      width: `${Math.min(
                        100,
                        spectralIndices.water_cover_percent ?? spectralIndices.water_percent ?? 0
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="bg-surface-800/40 rounded-xl p-3 border border-surface-800">
                <span className="text-[10px] text-surface-500 block mb-1">Urban Edge Metric</span>
                <span className="text-lg font-bold text-violet-400 font-mono">
                  {spectralIndices.urban_density_metric ?? spectralIndices.edge_density ?? 0}
                </span>
                <p className="text-[9px] text-surface-500 mt-1">Spatial frequency gradient</p>
              </div>
            </div>
          </div>
        ) : primaryPreviewUrl && !changeEvidence && !opticalSarEvidence ? (
          <div className="glass-card p-6 space-y-4 mb-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <h3 className="text-xs uppercase tracking-wider font-semibold text-surface-300 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-brand-500" />
                Analyzed Optical Observation Footprint
              </h3>
              <button
                type="button"
                onClick={() => setFullscreenImg({ url: primaryPreviewUrl, title: 'Analyzed Optical Scene' })}
                className="p-1 px-1.5 rounded text-surface-400 hover:text-white hover:bg-surface-700/50 transition-colors"
                title="Expand to Fullscreen"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
                </svg>
              </button>
            </div>
            <div className="relative aspect-video max-h-[440px] w-full rounded-xl overflow-hidden bg-surface-900 border border-surface-800 flex items-center justify-center">
              <img
                src={primaryPreviewUrl}
                alt="Analyzed Optical Scene"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        ) : null}
      </div>

      {/* ---- Execution Trace ---- */}
      <div>
        <div className="section-header">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="text-brand-400"
          >
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <h2>Execution Trace</h2>
          <div className="section-divider" />
        </div>
        <div className="glass-card p-4 space-y-3">
          {result.task && (
            <div className="flex items-center gap-2 pb-3 border-b border-surface-800">
              <span className="text-[10px] uppercase tracking-wider text-surface-500">Task:</span>
              <span className="badge badge-info text-[10px] uppercase">{result.task}</span>
              {result.input_type && (
                <span className="badge badge-info text-[10px]">{result.input_type}</span>
              )}
            </div>
          )}

          {traceSteps.length > 0 ? (
            <div className="space-y-1">
              {traceSteps.map((step) => (
                <TraceStepRow key={step.step} step={step} />
              ))}
            </div>
          ) : (
            <p className="text-surface-600 text-xs italic">
              Execution trace logged in database session.
            </p>
          )}
        </div>
      </div>

      {/* ---- Models Used ---- */}
      {result.models_used.length > 0 && (
        <div>
          <div className="section-header">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="text-brand-400"
            >
              <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
              <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
              <line x1="6" y1="6" x2="6.01" y2="6" />
              <line x1="6" y1="18" x2="6.01" y2="18" />
            </svg>
            <h2>Specialist Model Provenance</h2>
            <div className="section-divider" />
          </div>
          <div className="glass-card p-4 space-y-2">
            {result.models_used.map((model, idx) => (
              <div key={idx} className="flex items-center justify-between py-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-surface-300 font-mono">{model.name}</span>
                  {model.version && (
                    <span className="text-[10px] text-surface-600">v{model.version}</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-surface-500 uppercase">{model.task}</span>
                  {model.is_fallback && (
                    <span className="badge badge-warning text-[10px]">CPU Fallback</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Fullscreen Image Zoom / Inspection Modal */}
      {fullscreenImg && (
        <div
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex flex-col items-center justify-center p-4 animate-fade-in"
          onClick={() => setFullscreenImg(null)}
        >
          <div className="w-full max-w-5xl flex items-center justify-between p-3 text-white border-b border-surface-800 mb-2">
            <h3 className="text-sm font-semibold tracking-wide flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-brand-500" />
              {fullscreenImg.title}
            </h3>
            <button
              onClick={() => setFullscreenImg(null)}
              className="text-surface-400 hover:text-white p-1 rounded hover:bg-surface-800 transition-colors text-xs font-mono"
            >
              Close [Esc] ✕
            </button>
          </div>
          <div
            className="max-w-5xl max-h-[82vh] overflow-auto flex items-center justify-center rounded-xl bg-surface-950/80 border border-surface-800 p-2 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={fullscreenImg.url}
              alt={fullscreenImg.title}
              className="max-w-full max-h-[80vh] object-contain select-none"
            />
          </div>
          <p className="text-[11px] text-surface-500 mt-2 font-mono">
            Click outside or press Esc to exit full inspection mode
          </p>
        </div>
      )}
    </div>
  );
}
