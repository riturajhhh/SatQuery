/**
 * SatQuery AI — Header component.
 *
 * Top navigation bar with brand, health status indicator, and system info.
 */

import { useState, useEffect } from 'react';
import { checkHealth } from '../services/api';
import type { HealthStatus } from '../types';

interface HeaderProps {
  onOpenHistory?: () => void;
  onOpenModels?: () => void;
}

export default function Header({ onOpenHistory, onOpenModels }: HeaderProps) {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState(false);


  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await checkHealth();
        setHealth(data);
        setHealthError(false);
      } catch {
        setHealthError(true);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-50 glass-card rounded-none border-x-0 border-t-0">
      <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shadow-glow">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                <path d="M2 12h20" />
              </svg>
            </div>
            <div className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-accent-emerald border-2 border-surface-950 animate-pulse-soft" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              <span className="text-gradient-brand">SatQuery</span>
              <span className="text-surface-300 font-light ml-1">AI</span>
            </h1>
            <p className="text-[10px] text-surface-500 uppercase tracking-[0.2em] font-medium -mt-0.5">
              Remote-Sensing Intelligence
            </p>
          </div>
        </div>

        {/* System Status & History */}
        <div className="flex items-center gap-3">
          {onOpenModels && (
            <button
              onClick={onOpenModels}
              className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1.5 hover:border-brand-500/50"
              title="View loaded specialist models and registry"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
                <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
                <line x1="6" y1="6" x2="6.01" y2="6" />
                <line x1="6" y1="18" x2="6.01" y2="18" />
              </svg>
              Model Registry
            </button>
          )}

          {onOpenHistory && (
            <button
              onClick={onOpenHistory}
              className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1.5 hover:border-brand-500/50"
              title="Browse previous analysis sessions and reports"
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              History & Audits
            </button>
          )}

          {/* Health Status */}
          <div className="flex items-center gap-2">
            {health && (
              <>
                <div className={`badge ${health.gpu_available ? 'badge-success' : 'badge-warning'}`}>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <rect x="4" y="4" width="16" height="16" rx="2" />
                    <rect x="8" y="8" width="8" height="8" rx="1" />
                  </svg>
                  {health.gpu_available ? 'GPU' : 'CPU'}
                </div>
                <div className={`badge ${health.status === 'healthy' ? 'badge-success' : 'badge-error'}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${health.status === 'healthy' ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                  {health.status === 'healthy' ? 'Online' : 'Degraded'}
                </div>
              </>
            )}
            {healthError && (
              <div className="badge badge-error">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
                Offline
              </div>
            )}
            {!health && !healthError && (
              <div className="badge badge-info">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse-soft" />
                Connecting...
              </div>
            )}
          </div>

          {/* Version */}
          {health && (
            <span className="text-xs text-surface-600 font-mono hidden sm:inline">
              v{health.version}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}

