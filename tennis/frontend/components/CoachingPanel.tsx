'use client';

import { useState } from 'react';
import { MessageSquare, Volume2, VolumeX, Lightbulb, AlertTriangle, ChevronRight } from 'lucide-react';
import clsx from 'clsx';
import type { FrameOverlay } from './VideoPlayer';

interface CoachingPanelProps {
    activeOverlay: FrameOverlay | null;
    voiceEnabled: boolean;
    onToggleVoice: () => void;
    onSpeakText: (text: string) => void;
    overallRating?: string;
    overallScore?: number;
    topIssue?: string;
}

const SEVERITY_STYLES = {
    high: { border: 'border-red-500/30', bg: 'bg-red-500/10', text: 'text-red-400', icon: AlertTriangle },
    medium: { border: 'border-amber-500/30', bg: 'bg-amber-500/10', text: 'text-amber-400', icon: Lightbulb },
    low: { border: 'border-blue-500/30', bg: 'bg-blue-500/10', text: 'text-blue-400', icon: MessageSquare },
    info: { border: 'border-emerald-500/30', bg: 'bg-emerald-500/10', text: 'text-emerald-400', icon: MessageSquare },
};

export function CoachingPanel({
    activeOverlay,
    voiceEnabled,
    onToggleVoice,
    onSpeakText,
    overallRating = 'Intermediate',
    overallScore = 6.5,
    topIssue,
}: CoachingPanelProps) {
    const [showTip, setShowTip] = useState(false);

    const severity = activeOverlay?.severity ?? 'medium';
    const styles = SEVERITY_STYLES[severity as keyof typeof SEVERITY_STYLES] ?? SEVERITY_STYLES.medium;
    const Icon = styles.icon;

    return (
        <div className="h-full flex flex-col gap-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center">
                        <MessageSquare className="w-4 h-4 text-violet-400" />
                    </div>
                    <div>
                        <p className="text-sm font-semibold text-white">AI Coach</p>
                        <p className="text-xs text-white/40">{overallRating} · {overallScore}/10</p>
                    </div>
                </div>
                <button
                    onClick={onToggleVoice}
                    className={clsx(
                        'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all',
                        voiceEnabled
                            ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                            : 'bg-white/5 text-white/40 border border-white/10 hover:border-white/20',
                    )}
                    title={voiceEnabled ? 'Mute voice coach' : 'Enable voice coach'}
                >
                    {voiceEnabled ? <Volume2 className="w-3 h-3" /> : <VolumeX className="w-3 h-3" />}
                    {voiceEnabled ? 'Voice On' : 'Voice Off'}
                </button>
            </div>

            {/* Active coaching moment */}
            {activeOverlay ? (
                <div className={clsx(
                    'rounded-xl border p-4 transition-all duration-300',
                    styles.border, styles.bg,
                )}>
                    <div className="flex items-start gap-2.5 mb-3">
                        <Icon className={clsx('w-4 h-4 mt-0.5 flex-shrink-0', styles.text)} />
                        <div>
                            <p className={clsx('text-sm font-semibold mb-1', styles.text)}>
                                {activeOverlay.label}
                            </p>
                            <p className="text-sm text-white/70 leading-relaxed">
                                {activeOverlay.coaching_text}
                            </p>
                        </div>
                    </div>

                    {/* Tip toggle */}
                    {activeOverlay.tip && (
                        <div className="mt-2">
                            <button
                                onClick={() => setShowTip(!showTip)}
                                className="flex items-center gap-1 text-xs text-white/40 hover:text-white/60 transition-colors"
                            >
                                <ChevronRight className={clsx('w-3 h-3 transition-transform', showTip && 'rotate-90')} />
                                Quick fix
                            </button>
                            {showTip && (
                                <p className="mt-1.5 text-xs text-white/60 pl-4 border-l border-white/10 leading-relaxed">
                                    {activeOverlay.tip}
                                </p>
                            )}
                        </div>
                    )}

                    {/* Speak button */}
                    <button
                        onClick={() => onSpeakText(activeOverlay.coaching_text ?? activeOverlay.label)}
                        className="mt-3 flex items-center gap-1.5 text-xs text-white/40 hover:text-violet-400 transition-colors"
                    >
                        <Volume2 className="w-3 h-3" />
                        Speak coaching point
                    </button>
                </div>
            ) : (
                // Idle state
                <div className="flex-1 rounded-xl border border-white/5 bg-white/2 p-4 flex flex-col items-center justify-center text-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-violet-500/10 flex items-center justify-center">
                        <Lightbulb className="w-5 h-5 text-violet-400 opacity-50" />
                    </div>
                    <div>
                        <p className="text-sm text-white/40">Watching for coaching moments…</p>
                        <p className="text-xs text-white/25 mt-1">Coaching markers appear as you play through the video</p>
                    </div>
                </div>
            )}

            {/* Persistent top issue reminder */}
            {topIssue && !activeOverlay && (
                <div className="rounded-lg border border-white/5 bg-white/3 p-3">
                    <p className="text-xs text-white/30 uppercase tracking-wider mb-1">Primary focus</p>
                    <p className="text-sm text-white/60">{topIssue}</p>
                </div>
            )}
        </div>
    );
}
