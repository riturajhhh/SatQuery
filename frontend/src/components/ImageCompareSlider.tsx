import React, { useState, useRef, useCallback, useEffect } from 'react';

interface ImageCompareSliderProps {
  beforeImage: string;
  afterImage: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export default function ImageCompareSlider({
  beforeImage,
  afterImage,
  beforeLabel = 'Before (T1)',
  afterLabel = 'After (T2)',
  className = '',
}: ImageCompareSliderProps) {
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(percentage);
  }, []);

  const handleTouchMove = useCallback(
    (e: TouchEvent) => {
      if (!isDragging) return;
      handleMove(e.touches[0].clientX);
    },
    [isDragging, handleMove]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return;
      handleMove(e.clientX);
    },
    [isDragging, handleMove]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setSliderPos((prev) => Math.max(0, prev - 5));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setSliderPos((prev) => Math.min(100, prev + 5));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setSliderPos(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      setSliderPos(100);
    }
  };

  return (
    <div
      ref={containerRef}
      className={`relative w-full overflow-hidden rounded-xl bg-surface-900 border border-surface-800 select-none cursor-ew-resize group ${className}`}
      onMouseDown={(e) => {
        setIsDragging(true);
        handleMove(e.clientX);
      }}
      onTouchStart={(e) => {
        setIsDragging(true);
        handleMove(e.touches[0].clientX);
      }}
      tabIndex={0}
      role="slider"
      aria-valuenow={Math.round(sliderPos)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="Image comparison before and after swipe slider"
      onKeyDown={handleKeyDown}
      style={{ outline: 'none' }}
    >
      {/* Underlying Image (After / Right) */}
      <img
        src={afterImage}
        alt={afterLabel}
        className="w-full h-auto max-h-[460px] object-contain pointer-events-none block"
        draggable={false}
      />

      {/* Overlaid Image (Before / Left) with Clip Path */}
      <div
        className="absolute inset-0 overflow-hidden pointer-events-none"
        style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
      >
        <img
          src={beforeImage}
          alt={beforeLabel}
          className="w-full h-full object-contain pointer-events-none block"
          draggable={false}
        />
      </div>

      {/* Floating Labels */}
      <div className="absolute top-3 left-3 pointer-events-none z-10">
        <span className="badge bg-surface-900/90 text-cyan-300 border border-cyan-500/30 text-[10px] backdrop-blur font-medium shadow-md">
          {beforeLabel}
        </span>
      </div>
      <div className="absolute top-3 right-3 pointer-events-none z-10">
        <span className="badge bg-surface-900/90 text-amber-300 border border-amber-500/30 text-[10px] backdrop-blur font-medium shadow-md">
          {afterLabel}
        </span>
      </div>

      {/* Dividing Line & Grab Handle */}
      <div
        className="absolute top-0 bottom-0 z-20 pointer-events-none flex items-center justify-center"
        style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}
      >
        {/* Vertical Line */}
        <div className="w-[2px] h-full bg-white shadow-[0_0_8px_rgba(34,211,238,0.8)]" />

        {/* Grab Handle */}
        <div className="absolute w-8 h-8 rounded-full bg-surface-900 border-2 border-brand-400 flex items-center justify-center text-brand-300 shadow-glow transition-transform group-hover:scale-110">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="15 18 9 12 15 6" />
            <polyline points="9 18 3 12 9 6" />
          </svg>
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="-ml-1"
          >
            <polyline points="9 18 15 12 9 6" />
            <polyline points="15 18 21 12 15 6" />
          </svg>
        </div>

        {/* Position Pill */}
        <div className="absolute -bottom-1 bg-surface-950/90 border border-surface-700 text-[9px] font-mono text-surface-300 px-1.5 py-0.5 rounded shadow">
          {Math.round(sliderPos)}%
        </div>
      </div>
    </div>
  );
}
