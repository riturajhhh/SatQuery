/**
 * SatQuery AI — Query Input component.
 *
 * Natural-language query input with categorized demo query suggestions
 * and sensor requirement guidance chips.
 */

import { useState } from 'react';

export interface EnrichedDemoQuery {
  label: string;
  query: string;
  category: 'single' | 'change' | 'optical_sar' | 'isro';
  icon: string;
  requirement: string;
  badgeClass: string;
}

const DEMO_QUERIES: EnrichedDemoQuery[] = [
  {
    label: 'Land Cover Description',
    query: 'Describe the land-cover, vegetation, and major human-made objects visible in this satellite scene.',
    category: 'single',
    icon: '🌍',
    requirement: '1 Optical Image',
    badgeClass: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  },
  {
    label: 'Runway & Aircraft Grounding',
    query: 'Locate and pinpoint all airport runways and aircraft in the image with spatial bounding coordinates.',
    category: 'single',
    icon: '🎯',
    requirement: '1 Optical Image',
    badgeClass: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
  },
  {
    label: '🇮🇳 Cartosat-2S High-Res',
    query: 'Analyze high-resolution 0.65m features to detect built structures and road networks in Cartosat-2S imagery.',
    category: 'isro',
    icon: '🛰️',
    requirement: 'Cartosat-2S / High-Res',
    badgeClass: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
  },
  {
    label: '🇮🇳 RISAT-1A C-band SAR',
    query: 'Analyze C-band SAR backscatter anomalies to identify water bodies and flooded vegetation regardless of cloud cover.',
    category: 'isro',
    icon: '📡',
    requirement: 'RISAT-1A / SAR',
    badgeClass: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
  },
  {
    label: 'Surface Shift Detection',
    query: 'What changed between these two dates, and where did the change occur?',
    category: 'change',
    icon: '🔄',
    requirement: '2 Temporal Images',
    badgeClass: 'text-rose-400 bg-rose-500/10 border-rose-500/30',
  },
  {
    label: 'Urban Built-up Change',
    query: 'Has the built-up area increased, decreased, or remained unchanged between the baseline and follow-up scenes?',
    category: 'change',
    icon: '🏗️',
    requirement: '2 Temporal Images',
    badgeClass: 'text-rose-400 bg-rose-500/10 border-rose-500/30',
  },
  {
    label: 'Optical + SAR Fusion',
    query: 'Use the Cartosat optical and RISAT SAR images together to identify built-up and water-covered regions through cloud layers.',
    category: 'optical_sar',
    icon: '⚡',
    requirement: 'Optical + SAR Pair',
    badgeClass: 'text-sky-400 bg-sky-500/10 border-sky-500/30',
  },
];

type CategoryFilter = 'all' | 'single' | 'change' | 'optical_sar' | 'isro';

const CATEGORY_TABS: { id: CategoryFilter; label: string }[] = [
  { id: 'all', label: 'All Queries' },
  { id: 'single', label: 'Single Optical' },
  { id: 'change', label: 'Bi-Temporal' },
  { id: 'optical_sar', label: 'Optical + SAR' },
  { id: 'isro', label: '🇮🇳 ISRO / SAC' },
];

interface QueryInputProps {
  onSubmit: (query: string) => void;
  disabled?: boolean;
  isLoading?: boolean;
}

export default function QueryInput({ onSubmit, disabled = false, isLoading = false }: QueryInputProps) {
  const [query, setQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState<CategoryFilter>('all');
  const [activeTip, setActiveTip] = useState<string | null>(null);

  const handleSubmit = () => {
    if (query.trim() && !disabled && !isLoading) {
      onSubmit(query.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleDemoClick = (demoQuery: EnrichedDemoQuery) => {
    setQuery(demoQuery.query);
    if (demoQuery.category === 'change') {
      setActiveTip('💡 Tip: Ensure "Bi-Temporal" image mode is selected above with Before & After dates loaded.');
    } else if (demoQuery.category === 'optical_sar') {
      setActiveTip('💡 Tip: Ensure "Optical + SAR" mode is selected above with both optical and radar images loaded.');
    } else {
      setActiveTip(null);
    }
  };

  const filteredQueries = DEMO_QUERIES.filter((q) =>
    activeCategory === 'all' ? true : q.category === activeCategory
  );

  return (
    <div className="space-y-4">
      {/* Section Header */}
      <div className="section-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-brand-400">
          <circle cx="11" cy="11" r="8" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <h2>Query</h2>
        <div className="section-divider" />
      </div>

      {/* Input */}
      <div className="relative">
        <textarea
          id="query-input"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            if (activeTip) setActiveTip(null);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the satellite image (e.g., detect changes, locate structures, or interpret cross-modal radar features)..."
          rows={3}
          disabled={disabled}
          className="input-field resize-none pr-14 text-sm"
        />
        <button
          id="analyze-button"
          onClick={handleSubmit}
          disabled={!query.trim() || disabled || isLoading}
          className="absolute bottom-3 right-3 w-10 h-10 rounded-lg bg-brand-500 flex items-center justify-center transition-all duration-200 hover:bg-brand-600 disabled:opacity-30 disabled:hover:bg-brand-500 shadow-md"
          title="Run Satellite Analysis"
        >
          {isLoading ? (
            <svg width="18" height="18" viewBox="0 0 24 24" className="animate-spin text-white">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" strokeDasharray="31.4 31.4" strokeLinecap="round" />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </div>

      {/* Recommended Input Guidance Tip */}
      {activeTip && (
        <div className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 px-3 py-2 rounded-lg flex items-center justify-between animate-fade-in">
          <span>{activeTip}</span>
          <button
            onClick={() => setActiveTip(null)}
            className="text-amber-400 hover:text-amber-200 ml-2 font-mono text-[10px]"
          >
            ✕
          </button>
        </div>
      )}

      {/* Demo Queries Categorization Tabs */}
      <div className="space-y-2.5">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <p className="text-[10px] uppercase tracking-widest text-surface-500 font-semibold">
            Recommended Queries by Workflow
          </p>
          <div className="flex items-center gap-1 bg-surface-900/80 p-0.5 rounded-lg border border-surface-800">
            {CATEGORY_TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveCategory(tab.id)}
                className={`text-[11px] px-2 py-0.5 rounded-md transition-all font-medium ${
                  activeCategory === tab.id
                    ? 'bg-brand-500 text-white shadow-sm'
                    : 'text-surface-400 hover:text-surface-200 hover:bg-surface-800/50'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Query Suggestion Chips */}
        <div className="flex flex-wrap gap-2">
          {filteredQueries.map((dq, idx) => (
            <button
              key={idx}
              onClick={() => handleDemoClick(dq)}
              className="group inline-flex items-center gap-2 px-3 py-1.5 text-xs text-surface-300 
                         bg-surface-800/40 border border-surface-700/50 rounded-lg
                         transition-all duration-200 hover:text-white hover:bg-surface-800 hover:border-brand-500/40"
            >
              <span className="text-sm">{dq.icon}</span>
              <span className="font-medium">{dq.label}</span>
              <span className={`text-[10px] px-1.5 py-0.2 rounded border font-mono ${dq.badgeClass}`}>
                {dq.requirement}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
