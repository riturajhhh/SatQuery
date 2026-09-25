/**
 * SatQuery AI — Upload Panel Component.
 *
 * Supports independent uploads for:
 * 1. Single Optical/Multispectral scenes (VQA, Object Grounding, Scene Description)
 * 2. Bi-Temporal Image Pairs with dedicated, separate slots for:
 *    - Image 1 (T1: Pre-Event / Baseline)
 *    - Image 2 (T2: Post-Event / Follow-up)
 * Includes independent drag-and-drop, separate file pickers, real-time geospatial
 * metadata extraction, temporal swapping, and bitemporal compatibility checks.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { uploadFiles, ApiError } from '../services/api';
import type { UploadResponse, FileInfo } from '../types';

interface UploadPanelProps {
  onFilesSelected: (files: File[]) => void;
  onUploadSuccess?: (response: UploadResponse | null) => void;
  maxFiles?: number;
}

type UploadMode = 'single' | 'bitemporal';

export default function UploadPanel({
  onFilesSelected,
  onUploadSuccess,
}: UploadPanelProps) {
  const [mode, setMode] = useState<UploadMode>('single');
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);

  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Independent drag states
  const [isDragActiveSingle, setIsDragActiveSingle] = useState(false);
  const [isDragActive1, setIsDragActive1] = useState(false);
  const [isDragActive2, setIsDragActive2] = useState(false);

  // Independent hidden file input refs
  const inputRefSingle = useRef<HTMLInputElement>(null);
  const inputRef1 = useRef<HTMLInputElement>(null);
  const inputRef2 = useRef<HTMLInputElement>(null);

  // Centralized upload handler whenever active files change
  const executeUpload = useCallback(
    async (filesToUpload: File[], currentMode: UploadMode) => {
      if (filesToUpload.length === 0) {
        setUploadResponse(null);
        onUploadSuccess?.(null);
        return;
      }

      setIsUploading(true);
      setUploadError(null);

      try {
        const inputType = currentMode === 'bitemporal' && filesToUpload.length === 2
          ? 'bi_temporal'
          : 'single';

        const response = await uploadFiles(filesToUpload, inputType);
        setUploadResponse(response);
        onUploadSuccess?.(response);
      } catch (err) {
        const message =
          err instanceof ApiError
            ? err.message
            : 'Failed to upload and process satellite imagery. Please check backend connection.';
        setUploadError(message);
        setUploadResponse(null);
        onUploadSuccess?.(null);
      } finally {
        setIsUploading(false);
      }
    },
    [onUploadSuccess]
  );

  // Effect to trigger sync when file1, file2, or mode changes
  useEffect(() => {
    let activeFiles: File[] = [];
    if (mode === 'single') {
      activeFiles = file1 ? [file1] : [];
    } else {
      activeFiles = [file1, file2].filter((f): f is File => f !== null);
    }
    onFilesSelected(activeFiles);
    executeUpload(activeFiles, mode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file1, file2, mode]);

  const handleModeChange = (newMode: UploadMode) => {
    if (newMode === mode) return;
    setMode(newMode);
    setUploadError(null);
  };

  // Handlers for Slot 1 (T1)
  const setSlot1 = (file: File | null) => {
    setFile1(file);
  };

  // Handlers for Slot 2 (T2)
  const setSlot2 = (file: File | null) => {
    setFile2(file);
  };

  // Swap T1 and T2
  const swapFiles = () => {
    if (!file1 && !file2) return;
    const temp = file1;
    setFile1(file2);
    setFile2(temp);
  };

  const clearAll = () => {
    setFile1(null);
    setFile2(null);
    setUploadResponse(null);
    setUploadError(null);
  };

  const formatSize = (bytes?: number | null): string => {
    if (!bytes) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getModalityBadge = (modality?: string | null) => {
    switch (modality?.toLowerCase()) {
      case 'sar':
        return <span className="badge badge-warning text-[10px]">📡 SAR</span>;
      case 'multispectral':
        return <span className="badge badge-success text-[10px]">🌈 Multispectral</span>;
      case 'optical':
      default:
        return <span className="badge badge-info text-[10px]">🛰️ Optical RGB</span>;
    }
  };

  // Helper to render uploaded image card
  const renderImageCard = (
    file: File,
    fileInfo: FileInfo | undefined,
    slotLabel: string,
    slotTag: string,
    themeColor: 'cyan' | 'purple' | 'brand',
    onRemove: () => void,
    onReplace: () => void
  ) => {
    const previewSrc =
      fileInfo?.preview_url || (file.type.startsWith('image/') ? URL.createObjectURL(file) : '');

    const borderClass =
      themeColor === 'cyan'
        ? 'border-cyan-500/40 hover:border-cyan-500/70 shadow-cyan-500/5'
        : themeColor === 'purple'
        ? 'border-purple-500/40 hover:border-purple-500/70 shadow-purple-500/5'
        : 'border-brand-500/40 hover:border-brand-500/70 shadow-brand-500/5';

    const badgeClass =
      themeColor === 'cyan'
        ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
        : themeColor === 'purple'
        ? 'bg-purple-500/20 text-purple-300 border-purple-500/30'
        : 'bg-brand-500/20 text-brand-300 border-brand-500/30';

    return (
      <div className={`glass-card p-3 relative group transition-all border ${borderClass}`}>
        {/* Action buttons header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5">
            <span className={`badge border text-[10px] font-bold uppercase tracking-wider ${badgeClass}`}>
              {slotTag}
            </span>
            <span className="text-[11px] text-surface-300 font-medium">{slotLabel}</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={onReplace}
              disabled={isUploading}
              className="text-[10px] px-2 py-0.5 rounded bg-surface-800/80 text-surface-300 hover:text-white hover:bg-surface-700 transition"
              title="Replace this image"
            >
              Change
            </button>
            <button
              type="button"
              onClick={onRemove}
              disabled={isUploading}
              className="w-5 h-5 rounded-full bg-surface-800/80 flex items-center justify-center text-surface-400 hover:text-white hover:bg-rose-500/80 transition"
              title="Remove file"
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        {/* Image Preview */}
        <div className="relative aspect-video rounded-lg overflow-hidden mb-2 bg-surface-900 border border-surface-800">
          {previewSrc ? (
            <img src={previewSrc} alt={slotLabel} className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-surface-600">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
              </svg>
            </div>
          )}
          <div className="absolute bottom-1.5 left-1.5 flex flex-wrap items-center gap-1">
            {fileInfo && getModalityBadge(fileInfo.modality)}
            {fileInfo?.metadata?.isro && (
              <span className="badge bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[9px] font-semibold">
                🇮🇳 {fileInfo.metadata.isro.platform}
              </span>
            )}
          </div>
        </div>

        {/* Metadata Details */}
        <div className="space-y-1">
          <p className="text-xs text-white font-medium truncate" title={file.name}>
            {file.name}
          </p>
          <div className="flex items-center justify-between text-[10px] text-surface-400">
            <span>{formatSize(fileInfo?.file_size_bytes ?? file.size)}</span>
            <span>{fileInfo?.format || 'Standard Raster'}</span>
          </div>

          {fileInfo && (
            <div className="pt-1.5 mt-1.5 border-t border-surface-800/80 grid grid-cols-2 gap-1 text-[9px]">
              <div className="bg-surface-800/40 rounded px-1.5 py-0.5">
                <span className="text-surface-500 block">Dims</span>
                <span className="font-mono text-surface-300">
                  {fileInfo.width && fileInfo.height ? `${fileInfo.width}×${fileInfo.height}` : '—'}
                  {fileInfo.bands ? ` (${fileInfo.bands}b)` : ''}
                </span>
              </div>
              <div className="bg-surface-800/40 rounded px-1.5 py-0.5">
                <span className="text-surface-500 block">CRS</span>
                <span className="font-mono text-surface-300 truncate block" title={fileInfo.crs || 'Unreferenced'}>
                  {fileInfo.crs || 'Unreferenced'}
                </span>
              </div>
              {fileInfo.resolution && (
                <div className="bg-surface-800/40 rounded px-1.5 py-0.5 col-span-2 flex items-center justify-between">
                  <span className="text-surface-500">Resolution</span>
                  <span className="font-mono text-surface-300">
                    {fileInfo.resolution.x.toFixed(2)}m × {fileInfo.resolution.y.toFixed(2)}m
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  // Helper to render empty drop zone for a slot
  const renderEmptySlot = (
    slotTag: string,
    slotTitle: string,
    slotSubtitle: string,
    themeColor: 'cyan' | 'purple' | 'brand',
    isDragActive: boolean,
    onDragOver: (e: React.DragEvent) => void,
    onDragLeave: () => void,
    onDrop: (e: React.DragEvent) => void,
    onClick: () => void
  ) => {
    const borderClass = isDragActive
      ? themeColor === 'cyan'
        ? 'border-cyan-400 bg-cyan-500/10'
        : 'border-purple-400 bg-purple-500/10'
      : themeColor === 'cyan'
      ? 'border-cyan-500/30 hover:border-cyan-400/60 bg-cyan-950/10'
      : 'border-purple-500/30 hover:border-purple-400/60 bg-purple-950/10';

    const iconColor =
      themeColor === 'cyan' ? 'text-cyan-400' : themeColor === 'purple' ? 'text-purple-400' : 'text-brand-400';

    const badgeClass =
      themeColor === 'cyan'
        ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
        : 'bg-purple-500/20 text-purple-300 border-purple-500/30';

    return (
      <div
        className={`relative flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl transition-all cursor-pointer text-center min-h-[190px] ${borderClass}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={onClick}
      >
        <span className={`badge border text-[10px] font-bold uppercase tracking-wider mb-2.5 ${badgeClass}`}>
          {slotTag}
        </span>

        <div className="w-10 h-10 rounded-xl bg-surface-800/80 flex items-center justify-center mb-2">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className={iconColor}>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>

        <p className="text-xs font-semibold text-white mb-0.5">{slotTitle}</p>
        <p className="text-[10px] text-surface-400 max-w-[170px]">{slotSubtitle}</p>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Section Header & Ingestion Status */}
      <div className="section-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-brand-400">
          <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
        <h2>Input Satellite Imagery</h2>
        {isUploading && (
          <span className="text-[11px] text-brand-400 animate-pulse font-mono ml-auto">
            Ingesting & Extracting Metadata...
          </span>
        )}
        <div className="section-divider" />
      </div>

      {/* Mode Selector Tabs */}
      <div className="flex rounded-xl bg-surface-900/90 p-1 border border-surface-800 gap-1">
        <button
          type="button"
          onClick={() => handleModeChange('single')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
            mode === 'single'
              ? 'bg-brand-500 text-white shadow-md shadow-brand-500/25'
              : 'text-surface-400 hover:text-surface-200 hover:bg-surface-800/40'
          }`}
        >
          <span>🛰️</span> Single Scene
        </button>
        <button
          type="button"
          onClick={() => handleModeChange('bitemporal')}
          className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
            mode === 'bitemporal'
              ? 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white shadow-md shadow-purple-500/25'
              : 'text-surface-400 hover:text-surface-200 hover:bg-surface-800/40'
          }`}
        >
          <span>⏱️</span> Bi-Temporal Pair (T1 & T2)
        </button>
      </div>

      {/* Error Alert */}
      {uploadError && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0 mt-0.5">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div className="flex-1">
            <span className="font-semibold">Upload Error: </span>
            {uploadError}
          </div>
        </div>
      )}

      {/* ---- BI-TEMPORAL MODE: 2 SEPARATE IMAGE SLOTS ---- */}
      {mode === 'bitemporal' && (
        <div className="space-y-3">
          {/* Sub-header with helper & Swap button */}
          <div className="flex items-center justify-between text-xs px-1">
            <span className="text-surface-400 text-[11px]">
              Add <strong className="text-cyan-400 font-semibold">T1 (Before)</strong> and <strong className="text-purple-400 font-semibold">T2 (After)</strong> separately:
            </span>
            {(file1 || file2) && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={swapFiles}
                  disabled={!file1 || !file2 || isUploading}
                  className="text-[11px] text-brand-400 hover:text-brand-300 flex items-center gap-1 font-medium bg-brand-500/10 px-2 py-0.5 rounded border border-brand-500/20 hover:bg-brand-500/20 transition disabled:opacity-40"
                  title="Swap T1 and T2 images"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M7 16V4M7 4L3 8M7 4L11 8M17 8v12M17 20l4-4M17 20l-4-4" />
                  </svg>
                  Swap T1 ⇄ T2
                </button>
                <button
                  type="button"
                  onClick={clearAll}
                  disabled={isUploading}
                  className="text-[11px] text-surface-500 hover:text-rose-400 transition"
                >
                  Clear Both
                </button>
              </div>
            )}
          </div>

          {/* Side-by-Side Dual Image Slots */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* ---- SLOT 1: Image 1 (T1) ---- */}
            <div>
              {file1 ? (
                renderImageCard(
                  file1,
                  uploadResponse?.files[0],
                  'Pre-Event Baseline',
                  'T1 • Before',
                  'cyan',
                  () => setSlot1(null),
                  () => inputRef1.current?.click()
                )
              ) : (
                renderEmptySlot(
                  'T1 • Pre-Event',
                  'Upload First Image (T1)',
                  'Earlier date / baseline scene (GeoTIFF, PNG, JPG)',
                  'cyan',
                  isDragActive1,
                  (e) => { e.preventDefault(); setIsDragActive1(true); },
                  () => setIsDragActive1(false),
                  (e) => {
                    e.preventDefault();
                    setIsDragActive1(false);
                    if (e.dataTransfer.files?.[0]) setSlot1(e.dataTransfer.files[0]);
                  },
                  () => inputRef1.current?.click()
                )
              )}
              <input
                ref={inputRef1}
                type="file"
                className="hidden"
                accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
                onChange={(e) => {
                  if (e.target.files?.[0]) setSlot1(e.target.files[0]);
                }}
              />
            </div>

            {/* ---- SLOT 2: Image 2 (T2) ---- */}
            <div>
              {file2 ? (
                renderImageCard(
                  file2,
                  uploadResponse?.files[uploadResponse.files.length > 1 ? 1 : 0],
                  'Post-Event Follow-up',
                  'T2 • After',
                  'purple',
                  () => setSlot2(null),
                  () => inputRef2.current?.click()
                )
              ) : (
                renderEmptySlot(
                  'T2 • Post-Event',
                  'Upload Second Image (T2)',
                  'Later date / change follow-up scene (GeoTIFF, PNG, JPG)',
                  'purple',
                  isDragActive2,
                  (e) => { e.preventDefault(); setIsDragActive2(true); },
                  () => setIsDragActive2(false),
                  (e) => {
                    e.preventDefault();
                    setIsDragActive2(false);
                    if (e.dataTransfer.files?.[0]) setSlot2(e.dataTransfer.files[0]);
                  },
                  () => inputRef2.current?.click()
                )
              )}
              <input
                ref={inputRef2}
                type="file"
                className="hidden"
                accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
                onChange={(e) => {
                  if (e.target.files?.[0]) setSlot2(e.target.files[0]);
                }}
              />
            </div>
          </div>

          {/* Temporal Status Connector / Compatibility Bar */}
          {file1 && file2 ? (
            <div
              className={`p-3 rounded-xl border text-xs flex items-start gap-2.5 transition-all ${
                uploadResponse?.compatibility.valid !== false
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                  : 'bg-amber-500/10 border-amber-500/20 text-amber-300'
              }`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0 mt-0.5">
                {uploadResponse?.compatibility.valid !== false ? (
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                ) : (
                  <circle cx="12" cy="12" r="10" />
                )}
                <polyline points="22 4 12 14.01 9 11.01" />
              </svg>
              <div className="flex-1">
                <div className="font-semibold mb-0.5">
                  Bi-Temporal Pair Ready (T1 ──▶ T2)
                </div>
                <p className="text-[11px] opacity-90">
                  {uploadResponse?.compatibility.message ||
                    'Both temporal observation layers are loaded and ready for Change Detection & Change VQA.'}
                </p>
              </div>
            </div>
          ) : (
            <div className="p-2.5 rounded-xl border border-surface-800 bg-surface-900/50 text-[11px] text-surface-400 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                </span>
                <span>
                  {!file1 && !file2
                    ? 'Upload both T1 and T2 images to run bi-temporal change analysis.'
                    : !file1
                    ? 'Please upload Image 1 (T1) in the left slot.'
                    : 'Please upload Image 2 (T2) in the right slot.'}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ---- SINGLE IMAGE MODE ---- */}
      {mode === 'single' && (
        <div className="space-y-3">
          {file1 ? (
            <div>
              {renderImageCard(
                file1,
                uploadResponse?.files[0],
                'Single Satellite Scene',
                'Input Scene',
                'brand',
                () => setFile1(null),
                () => inputRefSingle.current?.click()
              )}

              {/* Quick Prompt to switch to Bi-Temporal if needed */}
              <div className="mt-3 p-2.5 rounded-xl bg-surface-900/60 border border-surface-800 flex items-center justify-between text-xs text-surface-400">
                <span>Want to compare this with another date?</span>
                <button
                  type="button"
                  onClick={() => setMode('bitemporal')}
                  className="text-brand-400 hover:text-brand-300 font-medium hover:underline text-[11px] flex items-center gap-1"
                >
                  Switch to Bi-Temporal Pair ➔
                </button>
              </div>
            </div>
          ) : (
            <div
              id="upload-zone"
              className={`upload-zone ${isDragActiveSingle ? 'active' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragActiveSingle(true); }}
              onDragLeave={() => setIsDragActiveSingle(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragActiveSingle(false);
                if (e.dataTransfer.files?.[0]) setFile1(e.dataTransfer.files[0]);
              }}
              onClick={() => inputRefSingle.current?.click()}
            >
              <input
                ref={inputRefSingle}
                type="file"
                className="hidden"
                accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
                onChange={(e) => {
                  if (e.target.files?.[0]) setFile1(e.target.files[0]);
                }}
              />

              <div className="w-14 h-14 rounded-2xl bg-brand-500/10 flex items-center justify-center mb-3">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-brand-400">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>

              <p className="text-surface-300 font-medium text-sm mb-1">
                Drop satellite image here
              </p>
              <p className="text-surface-500 text-xs mb-3">
                GeoTIFF (.tif/.tiff), PNG, JPEG (up to 500MB)
              </p>

              <div className="flex items-center gap-3 text-[11px] text-surface-500">
                <span>VQA</span>
                <span>•</span>
                <span>Land Cover</span>
                <span>•</span>
                <span>Building Count</span>
                <span>•</span>
                <span>Grounding</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
