'use client';

import clsx from 'clsx';
import { CheckCircle2, Circle, Loader2, AlertCircle } from 'lucide-react';

export interface ProcessingStep {
    id: string;
    label: string;
    status: 'pending' | 'active' | 'complete' | 'error';
    detail?: string;
}

interface ProcessingStatusProps {
    steps: ProcessingStep[];
    progress?: number;         // 0-100
    estimatedSeconds?: number;
    className?: string;
}

export function ProcessingStatus({ steps, progress, estimatedSeconds, className }: ProcessingStatusProps) {
    return (
        <div className={clsx('card space-y-4', className)}>
            {/* Overall progress */}
            {progress !== undefined && (
                <div>
                    <div className="flex justify-between text-xs text-white/50 mb-2">
                        <span>Processing</span>
                        <span>
                            {Math.round(progress)}%
                            {estimatedSeconds !== undefined && estimatedSeconds > 0 && (
                                <> · ~{Math.ceil(estimatedSeconds / 60)} min remaining</>
                            )}
                        </span>
                    </div>
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                    </div>
                </div>
            )}

            {/* Steps */}
            <div className="space-y-3">
                {steps.map((step) => (
                    <div key={step.id} className="flex items-start gap-3">
                        <div className="mt-0.5 shrink-0">
                            {step.status === 'complete' && (
                                <CheckCircle2 className="w-5 h-5 text-court-400" />
                            )}
                            {step.status === 'active' && (
                                <Loader2 className="w-5 h-5 text-brand-400 animate-spin" />
                            )}
                            {step.status === 'pending' && (
                                <Circle className="w-5 h-5 text-white/20" />
                            )}
                            {step.status === 'error' && (
                                <AlertCircle className="w-5 h-5 text-red-400" />
                            )}
                        </div>
                        <div>
                            <p className={clsx(
                                'text-sm font-medium',
                                step.status === 'complete' && 'text-white/70',
                                step.status === 'active' && 'text-white',
                                step.status === 'pending' && 'text-white/30',
                                step.status === 'error' && 'text-red-400',
                            )}>
                                {step.label}
                            </p>
                            {step.detail && step.status === 'active' && (
                                <p className="text-xs text-white/40 mt-0.5">{step.detail}</p>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

// Helper — derive step list from backend state
export function deriveSteps(state: string, framesProcessed: number): ProcessingStep[] {
    const ALL_STEPS: Array<{ id: string; label: string; requiredState?: string }> = [
        { id: 'upload', label: 'Video uploaded' },
        { id: 'ball', label: 'Ball detection' },
        { id: 'player', label: 'Player tracking' },
        { id: 'pose', label: 'Pose estimation' },
        { id: 'rally', label: 'Rally segmentation' },
        { id: 'line', label: 'Line call analysis' },
        { id: 'speed', label: 'Shot speed measurement' },
        { id: 'stats', label: 'Statistics compilation' },
    ];

    const ORDER = ['upload', 'ball', 'player', 'pose', 'rally', 'line', 'speed', 'stats'];

    const isComplete = state === 'completed' || state === 'stopped';
    const isFailed = state === 'failed';

    // Use frame count as a rough proxy for progress (each step = 12.5%)
    const pseudoProgress = Math.min(8, Math.ceil((framesProcessed / Math.max(1, framesProcessed + 50)) * 8));
    let activeIdx = isComplete ? 8 : Math.min(7, pseudoProgress);
    if (isFailed) activeIdx = pseudoProgress;

    return ALL_STEPS.map((s, i) => ({
        id: s.id,
        label: s.label,
        status:
            isFailed && i === activeIdx
                ? 'error'
                : isComplete || i < activeIdx
                    ? 'complete'
                    : i === activeIdx
                        ? 'active'
                        : 'pending',
    }));
}
