'use client';

import { useEffect, useMemo } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { useAnalysisStatus } from '@/hooks/useAnalysisStatus';
import { Activity, CheckCircle2, Loader2, XCircle, Clock } from 'lucide-react';
import clsx from 'clsx';

export default function ProcessingPage() {
    const { id: jobId } = useParams<{ id: string }>();
    const router = useRouter();
    const { data: status, isLoading, isError, error, isNotFound } = useAnalysisStatus(jobId);

    // Auto-redirect when analysis finishes
    useEffect(() => {
        if (status?.state === 'complete') {
            const t = setTimeout(() => {
                router.push(`/results/${jobId}`);
            }, 1200);
            return () => clearTimeout(t);
        }
    }, [status?.state, jobId, router]);

    const stages = useMemo(
        () => status?.stages ?? DEFAULT_STAGES,
        [status],
    );

    const stageIdx = status?.stage_index ?? 0;
    const progress = status?.progress ?? 0;
    const isComplete = status?.state === 'complete';
    const isFailed = status?.state === 'failed';
    const isRunning = !isComplete && !isFailed;

    return (
        <div className="max-w-xl mx-auto px-4 py-16">
            {/* Stepper */}
            <div className="flex items-center gap-3 mb-10">
                {['Upload', 'Configure', 'Processing', 'Results'].map((step, i) => (
                    <div key={step} className="flex items-center gap-2">
                        {i > 0 && <div className="w-8 h-px bg-white/10" />}
                        <div className={clsx('step-dot', i < 2 ? 'step-dot-complete' : i === 2 ? 'step-dot-active' : 'step-dot-pending')}>
                            {i + 1}
                        </div>
                        <span className={clsx('text-xs font-medium', i <= 2 ? 'text-white' : 'text-white/30')}>{step}</span>
                    </div>
                ))}
            </div>

            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-brand-600/20 flex items-center justify-center">
                    {isComplete
                        ? <CheckCircle2 className="w-5 h-5 text-court-400" />
                        : (isFailed || isNotFound)
                            ? <XCircle className="w-5 h-5 text-red-400" />
                            : <Activity className="w-5 h-5 text-brand-400 animate-pulse" />}
                </div>
                <div>
                    <h1 className="text-2xl font-bold text-white">
                        {isComplete ? 'Analysis complete' : isNotFound ? 'Session expired' : isFailed ? 'Analysis failed' : 'Analysing your match'}
                    </h1>
                    <p className="text-white/40 text-sm">
                        {isComplete
                            ? 'Redirecting to results…'
                            : isNotFound
                                ? 'This analysis session no longer exists. The server may have been restarted.'
                                : isFailed
                                    ? (status?.error ?? 'An error occurred during analysis')
                                    : 'Processing your video — this takes seconds to minutes.'}
                    </p>
                </div>
            </div>

            {/* Loading connector */}
            {isLoading && (
                <div className="card text-white/40 text-sm text-center py-8 flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Connecting to analysis server…
                </div>
            )}

            {/* Job not found — clear action */}
            {isNotFound && (
                <div className="card border-amber-500/20 bg-amber-500/5 text-amber-300 text-sm mt-2">
                    <p className="mb-3">This analysis job was not found on the server. This usually happens when the backend was restarted.</p>
                    <a href="/upload" className="btn-primary inline-flex items-center gap-2 text-sm py-2 px-4">
                        Upload a new video →
                    </a>
                </div>
            )}

            {/* Generic network error (not 404) */}
            {isError && !isNotFound && (
                <div className="card border-red-500/20 bg-red-500/5 text-red-400 text-sm mt-2">
                    Cannot reach the backend. Make sure{' '}
                    <code className="text-xs bg-white/10 px-1 rounded">python -m uvicorn tennis.api.app:app --port 8000</code>{' '}
                    is running.
                    <p className="text-xs mt-1 text-white/30">{String(error)}</p>
                </div>
            )}

            {status && (
                <>
                    {/* Progress bar */}
                    <div className="card mb-4">
                        <div className="flex justify-between text-xs text-white/50 mb-3">
                            <span className="font-medium text-white/80">
                                {isComplete ? 'Done' : status.stage_name}
                            </span>
                            <span>{Math.round(progress)}%</span>
                        </div>
                        <div className="progress-bar">
                            <div
                                className={clsx(
                                    'progress-fill transition-all duration-700',
                                    isComplete && 'bg-court-400',
                                    isFailed && 'bg-red-500',
                                )}
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                    </div>

                    {/* Stage list */}
                    <div className="card divide-y divide-white/[0.04] mb-4">
                        {stages.map((stage, i) => {
                            const isDone = i < stageIdx || isComplete;
                            const isActive = i === stageIdx && isRunning;
                            const isPending = i > stageIdx && !isComplete;

                            return (
                                <div key={stage.id} className="flex items-center gap-3 py-3">
                                    <div className={clsx(
                                        'w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-xs',
                                        isDone ? 'bg-court-500/20 text-court-400' :
                                            isActive ? 'bg-brand-600/20 text-brand-400' :
                                                'bg-surface-700 text-white/20',
                                    )}>
                                        {isDone
                                            ? <CheckCircle2 className="w-3.5 h-3.5" />
                                            : isActive
                                                ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                : <span>{i + 1}</span>}
                                    </div>
                                    <span className={clsx(
                                        'text-sm',
                                        isDone ? 'text-white/70' :
                                            isActive ? 'text-white font-medium' :
                                                'text-white/30',
                                    )}>
                                        {stage.label}
                                    </span>
                                    {isDone && <span className="ml-auto text-xs text-court-500/60">done</span>}
                                    {isActive && <span className="ml-auto text-xs text-brand-400 animate-pulse">running</span>}
                                </div>
                            );
                        })}
                    </div>

                    {/* Live counters */}
                    <div className="grid grid-cols-3 gap-3">
                        {[
                            { label: 'Frames', value: status.frames_processed.toLocaleString() },
                            { label: 'Players', value: status.players_detected || '—' },
                            { label: 'Ball hits', value: status.ball_detections || '—' },
                        ].map((s) => (
                            <div key={s.label} className="card text-center py-4">
                                <div className="text-xl font-bold text-white">{s.value}</div>
                                <div className="text-xs text-white/40 mt-1">{s.label}</div>
                            </div>
                        ))}
                    </div>

                    {/* Duration estimate */}
                    {status.duration_seconds > 0 && isRunning && (
                        <div className="flex items-center gap-2 text-xs text-white/30 mt-4 justify-center">
                            <Clock className="w-3.5 h-3.5" />
                            Video: {Math.round(status.duration_seconds)}s · estimated analysis time: {estimateTime(status.duration_seconds)}
                        </div>
                    )}

                    {/* Complete callout */}
                    {isComplete && (
                        <div className="card border-court-500/20 bg-court-500/5 text-court-400 text-sm text-center mt-4 animate-fade-in">
                            ✓ Analysis complete — redirecting to your match report…
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

function estimateTime(durationSec: number): string {
    // ~1s of analysis per 10s of video (sampled processing)
    const estimate = Math.ceil(durationSec / 10);
    if (estimate < 60) return `~${estimate}s`;
    return `~${Math.ceil(estimate / 60)}m`;
}

const DEFAULT_STAGES = [
    { id: 'upload', label: 'Video uploaded' },
    { id: 'metadata', label: 'Reading video metadata' },
    { id: 'court', label: 'Detecting court layout' },
    { id: 'motion', label: 'Tracking player movement' },
    { id: 'ball', label: 'Tracking tennis ball' },
    { id: 'shots', label: 'Classifying shots' },
    { id: 'rallies', label: 'Segmenting rallies' },
    { id: 'stats', label: 'Compiling match statistics' },
];
