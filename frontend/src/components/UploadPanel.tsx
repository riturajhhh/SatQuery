/**
 * SatQuery AI — Upload Panel Component.
 *
 * Drag-and-drop file upload with real-time geospatial metadata extraction,
 * modality inference badges, web preview rendering, and pair compatibility check.
 */

import { useState, useCallback, useRef } from 'react';
import { uploadFiles, ApiError } from '../services/api';
import type { UploadResponse, FileInfo } from '../types';

interface UploadPanelProps {
  onFilesSelected: (files: File[]) => void;
  onUploadSuccess?: (response: UploadResponse | null) => void;
  maxFiles?: number;
}

export default function UploadPanel({
  onFilesSelected,
  onUploadSuccess,
  maxFiles = 2,
}: UploadPanelProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const performUpload = useCallback(
    async (fileList: File[]) => {
      if (fileList.length === 0) {
        setUploadResponse(null);
        onUploadSuccess?.(null);
        return;
      }

      setIsUploading(true);
      setUploadError(null);

      try {
        const response = await uploadFiles(fileList);
        setUploadResponse(response);
        onUploadSuccess?.(response);
      } catch (err) {
        const message =
          err instanceof ApiError
            ? err.message
            : 'Failed to upload and process satellite imagery. Please check the backend connection.';
        setUploadError(message);
        setUploadResponse(null);
        onUploadSuccess?.(null);
      } finally {
        setIsUploading(false);
      }
    },
    [onUploadSuccess]
  );

  const handleFiles = useCallback(
    (newFiles: FileList | null) => {
      if (!newFiles) return;

      const fileArray = Array.from(newFiles).slice(0, maxFiles);
      setSelectedFiles(fileArray);
      onFilesSelected(fileArray);
      performUpload(fileArray);
    },
    [maxFiles, onFilesSelected, performUpload]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragActive(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragActive(false);
  }, []);

  const removeFile = (index: number) => {
    const updated = selectedFiles.filter((_, i) => i !== index);
    setSelectedFiles(updated);
    onFilesSelected(updated);
    performUpload(updated);
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

  return (
    <div className="space-y-4">
      {/* Section Header */}
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
        <h2>Input Satellite Imagery</h2>
        {isUploading && (
          <span className="text-[11px] text-brand-400 animate-pulse font-mono ml-auto">
            Ingesting & Extracting Metadata...
          </span>
        )}
        <div className="section-divider" />
      </div>

      {/* Error Alert */}
      {uploadError && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-start gap-2">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="flex-shrink-0 mt-0.5"
          >
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

      {/* Upload Zone */}
      {selectedFiles.length === 0 ? (
        <div
          id="upload-zone"
          className={`upload-zone ${isDragActive ? 'active' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
            multiple
            onChange={(e) => handleFiles(e.target.files)}
          />

          <div className="w-16 h-16 rounded-2xl bg-brand-500/10 flex items-center justify-center mb-4">
            <svg
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              className="text-brand-400"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>

          <p className="text-surface-300 font-medium text-sm mb-1">
            Drop satellite imagery here
          </p>
          <p className="text-surface-500 text-xs">
            GeoTIFF (.tif/.tiff), PNG, JPEG — up to {maxFiles} files (max 500MB)
          </p>

          <div className="flex items-center gap-4 mt-4">
            <span className="text-xs text-surface-600">Single Image</span>
            <span className="text-xs text-surface-700">•</span>
            <span className="text-xs text-surface-600">Bi-Temporal Pair</span>
            <span className="text-xs text-surface-700">•</span>
            <span className="text-xs text-surface-600">Optical + SAR</span>
          </div>
        </div>
      ) : (
        /* File Previews & Geospatial Metadata */
        <div className="space-y-3">
          <div
            className={`grid gap-3 ${
              selectedFiles.length > 1 ? 'grid-cols-2' : 'grid-cols-1'
            }`}
          >
            {selectedFiles.map((file, idx) => {
              const fileInfo: FileInfo | undefined = uploadResponse?.files[idx];
              const previewSrc =
                fileInfo?.preview_url ||
                (file.type.startsWith('image/') ? URL.createObjectURL(file) : '');
              const label =
                selectedFiles.length === 1
                  ? 'Input Image'
                  : idx === 0
                  ? 'Image A (t1 / Primary)'
                  : 'Image B (t2 / Reference)';

              return (
                <div
                  key={idx}
                  className="glass-card p-3 relative group animate-fade-in"
                >
                  {/* Remove button */}
                  <button
                    onClick={() => removeFile(idx)}
                    disabled={isUploading}
                    className="absolute top-2 right-2 z-10 w-6 h-6 rounded-full bg-surface-800/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-rose-500/80"
                    title="Remove file"
                  >
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      className="text-white"
                    >
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>

                    {/* Image Preview */}
                    <div className="relative aspect-video rounded-lg overflow-hidden mb-3 bg-surface-900 border border-surface-800">
                      {previewSrc ? (
                        <img
                          src={previewSrc}
                          alt={label}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            className="text-surface-600"
                          >
                            <circle cx="12" cy="12" r="10" />
                            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                          </svg>
                        </div>
                      )}
                      <div className="absolute bottom-2 left-2 flex flex-wrap items-center gap-1.5">
                        <span className="badge badge-info text-[10px]">{label}</span>
                        {fileInfo && getModalityBadge(fileInfo.modality)}
                        {fileInfo?.metadata?.isro && (
                          <span className="badge bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[10px] font-semibold">
                            🇮🇳 {fileInfo.metadata.isro.platform}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Metadata Details */}
                    <div className="space-y-1.5">
                      <p
                        className="text-xs text-white font-medium truncate"
                        title={file.name}
                      >
                        {file.name}
                      </p>
                      <div className="flex items-center justify-between text-[10px] text-surface-500">
                        <span>{formatSize(fileInfo?.file_size_bytes ?? file.size)}</span>
                        <span>{fileInfo?.format || 'Standard Raster'}</span>
                      </div>

                      {fileInfo && (
                        <div className="pt-2 mt-2 border-t border-surface-800/80 grid grid-cols-2 gap-1.5 text-[10px]">
                          <div className="bg-surface-800/40 rounded px-2 py-1">
                            <span className="text-surface-500 block">Dimensions</span>
                            <span className="font-mono text-surface-300">
                              {fileInfo.width && fileInfo.height
                                ? `${fileInfo.width}×${fileInfo.height}`
                                : '—'}
                              {fileInfo.bands ? ` (${fileInfo.bands}b)` : ''}
                            </span>
                          </div>

                          <div className="bg-surface-800/40 rounded px-2 py-1">
                            <span className="text-surface-500 block">CRS</span>
                            <span
                              className="font-mono text-surface-300 truncate block"
                              title={fileInfo.metadata?.isro?.indian_crs_zone || fileInfo.crs || 'Non-georeferenced'}
                            >
                              {fileInfo.crs || (fileInfo.metadata?.isro ? 'ISRO Grid' : 'Unreferenced')}
                            </span>
                          </div>

                          {fileInfo.resolution && (
                            <div className="bg-surface-800/40 rounded px-2 py-1 col-span-2 flex items-center justify-between">
                              <span className="text-surface-500">Pixel Resolution</span>
                              <span className="font-mono text-surface-300">
                                {fileInfo.resolution.x.toFixed(2)}m ×{' '}
                                {fileInfo.resolution.y.toFixed(2)}m
                              </span>
                            </div>
                          )}

                          {fileInfo.metadata?.isro && (
                            <div className="bg-amber-500/10 border border-amber-500/20 rounded px-2 py-1 col-span-2 space-y-0.5">
                              <div className="flex items-center justify-between text-[10px]">
                                <span className="text-amber-400 font-medium">ISRO Sensor</span>
                                <span className="text-amber-200">{fileInfo.metadata.isro.sensor_type}</span>
                              </div>
                              <div className="flex items-center justify-between text-[9px] text-surface-400">
                                <span>Dyn. Range: {fileInfo.metadata.isro.bit_depth_effective}-bit</span>
                                <span>Pol: {fileInfo.metadata.isro.polarization}</span>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>


          {/* Compatibility Card (when 2 files uploaded) */}
          {uploadResponse && uploadResponse.files.length === 2 && (
            <div
              className={`p-3 rounded-xl border text-xs flex items-start gap-2.5 transition-all ${
                uploadResponse.compatibility.valid
                  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                  : 'bg-amber-500/10 border-amber-500/20 text-amber-300'
              }`}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="flex-shrink-0 mt-0.5"
              >
                {uploadResponse.compatibility.valid ? (
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                ) : (
                  <circle cx="12" cy="12" r="10" />
                )}
                {uploadResponse.compatibility.valid ? (
                  <polyline points="22 4 12 14.01 9 11.01" />
                ) : (
                  <line x1="12" y1="8" x2="12" y2="12" />
                )}
              </svg>
              <div className="flex-1">
                <div className="font-semibold mb-0.5">
                  {uploadResponse.compatibility.valid
                    ? 'Image Pair Validated'
                    : 'Pair Notice'}
                </div>
                <p className="text-[11px] opacity-90">
                  {uploadResponse.compatibility.message}
                </p>
              </div>
            </div>
          )}

          {/* Add Second Image Button */}
          {selectedFiles.length < maxFiles && (
            <button
              onClick={() => inputRef.current?.click()}
              disabled={isUploading}
              className="btn-secondary w-full text-xs py-2 flex items-center justify-center gap-2"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              Add Second Image (for Change / SAR Fusion)
            </button>
          )}

          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg"
            multiple
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      )}
    </div>
  );
}
