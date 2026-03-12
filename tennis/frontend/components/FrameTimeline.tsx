'use client';

import { AlertTriangle, Info, Zap, CheckCircle } from 'lucide-react';
import clsx from 'clsx';
import type { FrameOverlay } from './VideoPlayer';

interface FrameTimelineProps {
    overlays: FrameOverlay[];
    duration: number;
    currentTime: number;
    onSeek: (t: number) => void;
}

const SEVERITY_CONFIG = {
    high: { color: '#ef4444', bg: 'bg-red-500', icon: AlertTriangle, label: 'Critical' },
    medium: { color: '#f59e0b', bg: 'bg-amber-500', icon: Zap, label: 'Improve' },
    low: { color: '#3b82f6', bg: 'bg-blue-500', icon: Info, label: 'Tip' },
    info: { color: '#10b981', bg: 'bg-emerald-500', icon: CheckCircle, label: 'Info' },
};

export function FrameTimeline({ overlays, duration, currentTime, onSeek }: FrameTimelineProps) {
    if (!overlays.length || !duration) {
        return (
            <div className="w-full h-12 flex items-center justify-center text-white/30 text-xs">
                No coaching moments detected
            </div>
        );
    }

    return (
        <div className="w-full space-y-2">
            {/* Timeline bar */}
            <div className="relative h-8 bg-white/5 rounded-lg overflow-hidden">
                {/* Playhead */}
                <div
                    className="absolute top-0 bottom-0 w-0.5 bg-violet-400 z-10 opacity-80 transition-all"
                    style={{ left: `${(currentTime / duration) * 100}%` }}
                />
                {/* Moment markers */}
                {overlays.map((ov, i) => {
                    const cfg = SEVERITY_CONFIG[ov.severity ?? 'medium'];
                    const pct = (ov.timestamp / duration) * 100;
                    const isActive = currentTime >= ov.timestamp && currentTime <= ov.timestamp + (ov.duration ?? 2.5);
                    return (
                        <button
                            key={i}
                            className={clsx(
                                'absolute top-0 bottom-0 w-1.5 rounded-sm transition-all hover:w-2 cursor-pointer',
                                cfg.bg,
                                isActive ? 'opacity-100 scale-y-110' : 'opacity-60 hover:opacity-90',
                            )}
                            style={{ left: `${pct}%`, transform: 'translateX(-50%)' }}
                            onClick={() => onSeek(ov.timestamp)}
                            title={`${ov.label} — ${ov.timestamp.toFixed(1)}s`}
                        />
                    );
                })}
            </div>

            {/* Moment pills — scrollable list */}
            <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-hide">
                {overlays.map((ov, i) => {
                    const cfg = SEVERITY_CONFIG[ov.severity ?? 'medium'];
                    const Icon = cfg.icon;
                    const isActive = currentTime >= ov.timestamp && currentTime <= ov.timestamp + (ov.duration ?? 2.5);
                    return (
                        <button
                            key={i}
                            onClick={() => onSeek(ov.timestamp)}
                            className={clsx(
                                'flex-shrink-0 flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
                                'transition-all border',
                                isActive
                                    ? 'border-white/30 bg-white/10 text-white scale-105 shadow-md'
                                    : 'border-white/10 bg-white/5 text-white/50 hover:bg-white/10 hover:text-white/80',
                            )}
                        >
                            <Icon className="w-3 h-3" style={{ color: cfg.color }} />
                            <span>{ov.label}</span>
                            <span className="opacity-50">{ov.timestamp.toFixed(1)}s</span>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
