import { useEffect, useState } from 'react';
import { getAnalysisHistory, getReportUrl } from '../services/api';
import type { AnalysisHistoryItem } from '../types';

interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectAnalysis: (analysisId: string) => void;
}

export default function HistoryDrawer({ isOpen, onClose, onSelectAnalysis }: HistoryDrawerProps) {
  const [items, setItems] = useState<AnalysisHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      loadHistory();
    }
  }, [isOpen]);

  const loadHistory = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getAnalysisHistory(30, 0);
      setItems(data.items);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load analysis history');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm animate-fade-in">
      <div
        className="w-full max-w-md bg-surface-900 border-l border-surface-800 h-full flex flex-col shadow-2xl animate-slide-left"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Drawer Header */}
        <div className="p-4 border-b border-surface-800 flex items-center justify-between bg-surface-900/90 backdrop-blur">
          <div className="flex items-center gap-2">
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="text-brand-400"
            >
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            <h3 className="text-sm font-semibold text-white">Analysis History & Audits</h3>
            <span className="text-[11px] bg-surface-800 text-surface-400 px-2 py-0.5 rounded-full font-mono">
              {items.length}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={loadHistory}
              disabled={isLoading}
              className="text-surface-400 hover:text-white p-1 rounded hover:bg-surface-800 transition-colors"
              title="Refresh history"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className={isLoading ? 'animate-spin' : ''}
              >
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l6.73-6.73" />
              </svg>
            </button>
            <button
              onClick={onClose}
              className="text-surface-400 hover:text-white p-1 rounded hover:bg-surface-800 transition-colors"
              title="Close"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        {/* Drawer Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {isLoading && items.length === 0 && (
            <div className="py-16 text-center text-surface-400 text-xs space-y-2">
              <div className="w-6 h-6 border-2 border-brand-400 border-t-transparent rounded-full animate-spin mx-auto" />
              <p>Loading historical audit records...</p>
            </div>
          )}

          {error && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-300 text-xs">
              {error}
            </div>
          )}

          {!isLoading && items.length === 0 && !error && (
            <div className="py-16 text-center text-surface-500 text-xs space-y-2">
              <svg
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                className="mx-auto text-surface-600"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <line x1="9" y1="3" x2="9" y2="21" />
              </svg>
              <p className="font-medium text-surface-400">No previous sessions found</p>
              <p className="text-[11px]">Upload an image and run a query to record audit trails.</p>
            </div>
          )}

          {items.map((item) => {
            const dateStr = item.created_at
              ? new Date(item.created_at).toLocaleString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })
              : 'Unknown date';

            return (
              <div
                key={item.analysis_id}
                className="bg-surface-800/40 hover:bg-surface-800/70 border border-surface-800 rounded-lg p-3 transition-all space-y-2.5"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="text-[10px] bg-brand-500/10 text-brand-400 border border-brand-500/20 px-1.5 py-0.5 rounded font-mono">
                      {item.task?.toUpperCase() || 'QUERY'}
                    </span>
                    {item.modalities.map((m) => (
                      <span
                        key={m}
                        className="text-[10px] bg-surface-700/50 text-surface-300 px-1.5 py-0.5 rounded uppercase"
                      >
                        {m}
                      </span>
                    ))}
                  </div>
                  <span className="text-[10px] text-surface-500 font-mono flex-shrink-0">{dateStr}</span>
                </div>

                <p
                  className="text-xs text-white font-medium line-clamp-2 hover:text-brand-300 cursor-pointer"
                  onClick={() => {
                    onSelectAnalysis(item.analysis_id);
                    onClose();
                  }}
                  title="Click to view analysis"
                >
                  "{item.query}"
                </p>

                <div className="pt-2 border-t border-surface-800/60 flex items-center justify-between text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        item.status === 'complete' ? 'bg-emerald-400' : 'bg-amber-400'
                      }`}
                    />
                    <span className="text-surface-400 capitalize">{item.status}</span>
                    {item.confidence !== null && (
                      <span className="text-surface-500 text-[10px] font-mono">
                        ({Math.round(item.confidence * 100)}%)
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => {
                        onSelectAnalysis(item.analysis_id);
                        onClose();
                      }}
                      className="px-2 py-0.5 text-[10px] bg-surface-700 hover:bg-surface-600 text-surface-200 rounded font-medium transition-colors"
                    >
                      View
                    </button>
                    <a
                      href={getReportUrl(item.analysis_id, 'pdf')}
                      download={`satquery_report_${item.analysis_id.slice(0, 8)}.pdf`}
                      className="px-1.5 py-0.5 text-[10px] bg-brand-500/10 hover:bg-brand-500/20 text-brand-400 border border-brand-500/30 rounded font-medium transition-colors flex items-center gap-1"
                      title="Download PDF report"
                    >
                      PDF
                    </a>
                    <a
                      href={getReportUrl(item.analysis_id, 'html')}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-1.5 py-0.5 text-[10px] bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 rounded font-medium transition-colors flex items-center gap-1"
                      title="Open HTML report"
                    >
                      HTML
                    </a>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
