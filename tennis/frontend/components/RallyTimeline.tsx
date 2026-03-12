'use client';

import clsx from 'clsx';

interface RallyTimelineProps {
    rallies: RallyEntry[];
    activeId?: string | null;
    onSelect?: (rally: RallyEntry) => void;
}

export interface RallyEntry {
    id: string;
    number: number;
    startTime: number;      // seconds in video
    duration: number;       // seconds
    shotCount: number;
    winner: string;
    outcome: 'winner' | 'error' | 'ace' | 'fault';
    server: string;
}

const OUTCOME_MAP: Record<RallyEntry['outcome'], { label: string; cls: string }> = {
    winner: { label: 'Winner', cls: 'badge-success' },
    error: { label: 'Error', cls: 'badge-error' },
    ace: { label: 'Ace', cls: 'badge-brand' },
    fault: { label: 'Fault', cls: 'badge-warning' },
};

function formatTime(s: number) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

export function RallyTimeline({ rallies, activeId, onSelect }: RallyTimelineProps) {
    return (
        <div className="space-y-2 overflow-y-auto max-h-[480px] pr-1">
            {rallies.length === 0 && (
                <p className="text-white/40 text-sm text-center py-8">
                    No rallies detected yet.
                </p>
            )}
            {rallies.map((r) => {
                const outcome = OUTCOME_MAP[r.outcome];
                return (
                    <button
                        key={r.id}
                        onClick={() => onSelect?.(r)}
                        className={clsx(
                            'w-full text-left px-4 py-3 rounded-xl border transition-all duration-200',
                            activeId === r.id
                                ? 'bg-brand-600/15 border-brand-500/40'
                                : 'bg-surface-700/50 border-white/[0.06] hover:bg-surface-700 hover:border-white/10',
                        )}
                    >
                        <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-3">
                                <div className={clsx(
                                    'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold shrink-0',
                                    activeId === r.id ? 'bg-brand-600 text-white' : 'bg-surface-600 text-white/50',
                                )}>
                                    {r.number}
                                </div>
                                <div>
                                    <div className="text-sm font-medium text-white">{r.winner}</div>
                                    <div className="text-xs text-white/40">{r.shotCount} shots · {formatTime(r.startTime)}</div>
                                </div>
                            </div>
                            <span className={clsx('badge', outcome.cls)}>{outcome.label}</span>
                        </div>
                    </button>
                );
            })}
        </div>
    );
}
