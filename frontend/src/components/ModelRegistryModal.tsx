import { useEffect, useState } from 'react';
import { listModels } from '../services/api';
import type { ModelInfo } from '../types';

interface ModelRegistryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ModelRegistryModal({ isOpen, onClose }: ModelRegistryModalProps) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [gpuAvailable, setGpuAvailable] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedTask, setSelectedTask] = useState<string>('all');

  useEffect(() => {
    if (isOpen) {
      loadModels();
    }
  }, [isOpen]);

  const loadModels = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listModels();
      setModels(data.models);
      setGpuAvailable(data.gpu_available);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load model registry');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  const tasks = ['all', 'vqa', 'captioning', 'grounding', 'change_detection', 'optical_sar'];

  const filteredModels = models.filter((m) => {
    const matchesSearch =
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (m.base_model || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.supported_tasks.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesTask =
      selectedTask === 'all' || m.supported_tasks.some((t) => t.toLowerCase().includes(selectedTask.toLowerCase()));

    return matchesSearch && matchesTask;
  });

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-fade-in"
      onClick={onClose}
    >
      <div
        className="w-full max-w-3xl bg-surface-900 border border-surface-800 rounded-2xl flex flex-col max-h-[85vh] shadow-2xl overflow-hidden animate-scale-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="p-5 border-b border-surface-800 flex items-center justify-between bg-surface-900/90 backdrop-blur">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center text-brand-400">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                <line x1="8" y1="21" x2="16" y2="21" />
                <line x1="12" y1="17" x2="12" y2="21" />
              </svg>
            </div>
            <div>
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                Agentic Model Registry
                <span className="text-xs font-mono font-normal text-surface-400">({models.length} registered)</span>
              </h2>
              <p className="text-[11px] text-surface-400">
                Specialist Remote-Sensing Vision-Language Models & Fallback Engines
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span
              className={`badge text-[10px] ${
                gpuAvailable ? 'badge-success' : 'bg-amber-500/10 text-amber-300 border-amber-500/30'
              }`}
            >
              {gpuAvailable ? '⚡ GPU Acceleration' : '💻 CPU Optimized Inference'}
            </span>
            <button
              onClick={onClose}
              className="text-surface-400 hover:text-white p-1.5 rounded-lg hover:bg-surface-800 transition-colors"
              title="Close modal"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        {/* Filter Toolbar */}
        <div className="px-5 py-3 border-b border-surface-800/80 bg-surface-900/50 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="relative w-full sm:w-64">
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="absolute left-3 top-1/2 -translate-y-1/2 text-surface-500"
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              placeholder="Search models or tasks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 bg-surface-800/80 border border-surface-700/60 rounded-lg text-xs text-white placeholder-surface-500 focus:outline-none focus:border-brand-500"
            />
          </div>

          <div className="flex items-center gap-1 overflow-x-auto w-full sm:w-auto pb-1 sm:pb-0">
            {tasks.map((task) => (
              <button
                key={task}
                onClick={() => setSelectedTask(task)}
                className={`px-2.5 py-1 rounded-lg text-[10px] font-medium transition-colors uppercase whitespace-nowrap ${
                  selectedTask === task
                    ? 'bg-brand-500 text-white shadow-sm'
                    : 'bg-surface-800/60 text-surface-400 hover:text-surface-200'
                }`}
              >
                {task.replace('_', ' ')}
              </button>
            ))}
          </div>
        </div>

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {isLoading && models.length === 0 && (
            <div className="py-20 text-center text-surface-400 text-xs space-y-2">
              <div className="w-7 h-7 border-2 border-brand-400 border-t-transparent rounded-full animate-spin mx-auto" />
              <p>Querying Model Registry...</p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-300 text-xs">
              {error}
            </div>
          )}

          {!isLoading && filteredModels.length === 0 && !error && (
            <div className="py-16 text-center text-surface-500 text-xs">
              No models match your query.
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {filteredModels.map((m) => (
              <div
                key={m.name}
                className="bg-surface-800/40 hover:bg-surface-800/70 border border-surface-800 rounded-xl p-4 transition-all space-y-3"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-white">{m.name}</h3>
                    <p className="text-[11px] text-surface-400 font-mono mt-0.5">
                      Base: {m.base_model || 'Specialist Architecture'}
                    </p>
                  </div>
                  <span
                    className={`text-[9px] font-semibold px-2 py-0.5 rounded-full uppercase ${
                      m.is_fallback
                        ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                        : 'bg-brand-500/15 text-brand-300 border border-brand-500/30'
                    }`}
                  >
                    {m.is_fallback ? 'Fallback' : 'Primary'}
                  </span>
                </div>

                <div className="space-y-1 text-[11px]">
                  <div className="flex items-center justify-between text-surface-400">
                    <span>Hardware:</span>
                    <span className="font-mono text-surface-200">{m.device?.toUpperCase() || 'CPU'}</span>
                  </div>
                  {m.adapter && (
                    <div className="flex items-center justify-between text-surface-400">
                      <span>LoRA Adapter:</span>
                      <span className="font-mono text-brand-300">{m.adapter}</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between text-surface-400">
                    <span>Status:</span>
                    <span className="flex items-center gap-1 text-emerald-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                      Loaded & Ready
                    </span>
                  </div>
                </div>

                <div className="pt-2 border-t border-surface-800/80 flex flex-wrap gap-1">
                  {m.supported_tasks.map((task) => (
                    <span
                      key={task}
                      className="text-[9px] bg-surface-700/50 text-surface-300 px-1.5 py-0.5 rounded font-mono"
                    >
                      {task}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="p-4 border-t border-surface-800 bg-surface-900/90 flex items-center justify-between text-xs text-surface-500">
          <span>Self-healing fallback chain active for zero-crash operational continuity.</span>
          <button
            onClick={onClose}
            className="btn-secondary text-xs py-1 px-4 hover:border-brand-500/40"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
