'use client';

import { useState } from 'react';
import {
    Trophy, Target, Activity, ChevronDown, ChevronUp,
    AlertTriangle, Lightbulb, CheckCircle, TrendingUp, User2,
} from 'lucide-react';
import clsx from 'clsx';

// ─── Types ────────────────────────────────────────────────────────────────────

interface CoachingSection {
    title: string;
    scores?: Record<string, number>;
    narrative?: string;
    intensity?: number;
    speed_kmh?: number;
    patterns?: string[];
    crosscourt_rate?: number;
    net_rate?: number;
    score?: number;
    issues_detected?: number;
    coverage?: number[];
}

interface Recommendation {
    priority: 'high' | 'medium' | 'low';
    category: string;
    title: string;
    detail: string;
    drill?: string;
}

interface CoachingReportData {
    performance_summary: string;
    overall_rating: string;
    overall_score: number;
    sections: Record<string, CoachingSection>;
    recommendations: Recommendation[];
    strengths: string[];
    improvement_areas: string[];
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ScoreBar({ label, value }: { label: string; value: number }) {
    const pct = Math.min(100, (value / 10) * 100);
    const color = value >= 7.5 ? '#10b981' : value >= 5.5 ? '#f59e0b' : '#ef4444';
    return (
        <div className="flex items-center gap-3">
            <span className="text-xs text-white/50 w-20 flex-shrink-0 capitalize">{label}</span>
            <div className="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${pct}%`, background: color }}
                />
            </div>
            <span className="text-xs font-mono text-white/60 w-8 text-right">{value.toFixed(1)}</span>
        </div>
    );
}

function PriorityBadge({ priority }: { priority: string }) {
    const config = {
        high: { bg: 'bg-red-500/15', text: 'text-red-400', label: 'Critical' },
        medium: { bg: 'bg-amber-500/15', text: 'text-amber-400', label: 'Improve' },
        low: { bg: 'bg-blue-500/15', text: 'text-blue-400', label: 'Tip' },
    }[priority] ?? { bg: 'bg-white/10', text: 'text-white/50', label: priority };
    return (
        <span className={clsx('px-2 py-0.5 rounded-full text-xs font-medium', config.bg, config.text)}>
            {config.label}
        </span>
    );
}

function CollapsibleSection({ title, icon: Icon, children }: {
    title: string;
    icon: React.ElementType;
    children: React.ReactNode;
}) {
    const [open, setOpen] = useState(true);
    return (
        <div className="border border-white/5 rounded-xl overflow-hidden">
            <button
                className="w-full flex items-center justify-between px-4 py-3 bg-white/3 hover:bg-white/5 transition-colors"
                onClick={() => setOpen(!open)}
            >
                <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4 text-violet-400" />
                    <span className="text-sm font-semibold text-white">{title}</span>
                </div>
                {open ? <ChevronUp className="w-4 h-4 text-white/30" /> : <ChevronDown className="w-4 h-4 text-white/30" />}
            </button>
            {open && <div className="px-4 py-4">{children}</div>}
        </div>
    );
}

// ─── Main CoachingReport ──────────────────────────────────────────────────────

export function CoachingReport({ report }: { report: CoachingReportData }) {
    const {
        performance_summary,
        overall_rating,
        overall_score,
        sections = {},
        recommendations = [],
        strengths = [],
        improvement_areas = [],
    } = report;

    const ratingColors: Record<string, string> = {
        Elite: '#10b981', Advanced: '#8b5cf6', Intermediate: '#f59e0b', Beginner: '#ef4444',
    };

    return (
        <div className="space-y-4">
            {/* Performance header */}
            <div className="rounded-xl bg-gradient-to-br from-violet-500/10 to-purple-500/5 border border-violet-500/20 p-5">
                <div className="flex items-start justify-between mb-3">
                    <div>
                        <div className="flex items-center gap-2 mb-1">
                            <span
                                className="text-sm font-bold px-2 py-0.5 rounded-full"
                                style={{
                                    background: `${ratingColors[overall_rating] ?? '#8b5cf6'}20`,
                                    color: ratingColors[overall_rating] ?? '#8b5cf6',
                                }}
                            >
                                {overall_rating}
                            </span>
                            <span className="text-xl font-bold text-white">{overall_score.toFixed(1)}<span className="text-sm text-white/40">/10</span></span>
                        </div>
                        <Trophy className="w-5 h-5 text-amber-400 mb-1" />
                    </div>
                </div>
                <p className="text-sm text-white/60 leading-relaxed">{performance_summary}</p>
            </div>

            {/* Shot quality */}
            {sections.shot_quality?.scores && (
                <CollapsibleSection title="Shot Quality" icon={Target}>
                    <div className="space-y-2.5 mb-3">
                        {Object.entries(sections.shot_quality.scores).map(([k, v]) => (
                            <ScoreBar key={k} label={k} value={v as number} />
                        ))}
                    </div>
                    {sections.shot_quality.narrative && (
                        <p className="text-xs text-white/40 leading-relaxed">{sections.shot_quality.narrative}</p>
                    )}
                </CollapsibleSection>
            )}

            {/* Movement */}
            {sections.movement && (
                <CollapsibleSection title="Movement & Footwork" icon={Activity}>
                    <div className="grid grid-cols-2 gap-3 mb-3">
                        {sections.movement.speed_kmh !== undefined && (
                            <div className="bg-white/5 rounded-lg p-3 text-center">
                                <p className="text-xl font-bold text-white">{(sections.movement.speed_kmh as number).toFixed(1)}</p>
                                <p className="text-xs text-white/40">Avg speed km/h</p>
                            </div>
                        )}
                        {sections.movement.intensity !== undefined && (
                            <div className="bg-white/5 rounded-lg p-3 text-center">
                                <p className="text-xl font-bold text-white">{((sections.movement.intensity as number) * 100).toFixed(0)}%</p>
                                <p className="text-xs text-white/40">Movement intensity</p>
                            </div>
                        )}
                    </div>
                    {sections.movement.narrative && (
                        <p className="text-xs text-white/40 leading-relaxed">{sections.movement.narrative}</p>
                    )}
                </CollapsibleSection>
            )}

            {/* Tactics */}
            {sections.tactics && (
                <CollapsibleSection title="Tactical Analysis" icon={TrendingUp}>
                    {sections.tactics.patterns && (
                        <div className="flex flex-wrap gap-2 mb-3">
                            {(sections.tactics.patterns as string[]).map((p, i) => (
                                <span key={i} className="text-xs px-2 py-1 rounded-full bg-violet-500/10 text-violet-300 border border-violet-500/20">
                                    {p}
                                </span>
                            ))}
                        </div>
                    )}
                    <div className="grid grid-cols-2 gap-3 mb-3">
                        {sections.tactics.crosscourt_rate !== undefined && (
                            <div className="bg-white/5 rounded-lg p-3 text-center">
                                <p className="text-xl font-bold text-white">{((sections.tactics.crosscourt_rate as number) * 100).toFixed(0)}%</p>
                                <p className="text-xs text-white/40">Crosscourt rate</p>
                            </div>
                        )}
                        {sections.tactics.net_rate !== undefined && (
                            <div className="bg-white/5 rounded-lg p-3 text-center">
                                <p className="text-xl font-bold text-white">{((sections.tactics.net_rate as number) * 100).toFixed(0)}%</p>
                                <p className="text-xs text-white/40">Net approaches</p>
                            </div>
                        )}
                    </div>
                    {sections.tactics.narrative && (
                        <p className="text-xs text-white/40 leading-relaxed">{sections.tactics.narrative}</p>
                    )}
                </CollapsibleSection>
            )}

            {/* Strengths */}
            {strengths.length > 0 && (
                <CollapsibleSection title="Strengths" icon={CheckCircle}>
                    <ul className="space-y-2">
                        {strengths.map((s, i) => (
                            <li key={i} className="flex items-start gap-2 text-sm text-white/60">
                                <CheckCircle className="w-3.5 h-3.5 text-emerald-400 mt-0.5 flex-shrink-0" />
                                {s}
                            </li>
                        ))}
                    </ul>
                </CollapsibleSection>
            )}

            {/* Recommendations */}
            {recommendations.length > 0 && (
                <CollapsibleSection title="Recommendations" icon={Lightbulb}>
                    <div className="space-y-3">
                        {recommendations.map((rec, i) => (
                            <div key={i} className="rounded-lg border border-white/5 bg-white/3 p-3">
                                <div className="flex items-center gap-2 mb-1.5">
                                    <PriorityBadge priority={rec.priority} />
                                    <span className="text-sm font-medium text-white">{rec.title}</span>
                                </div>
                                <p className="text-xs text-white/50 leading-relaxed mb-1">{rec.detail}</p>
                                {rec.drill && (
                                    <p className="text-xs text-violet-400/70">{rec.drill}</p>
                                )}
                            </div>
                        ))}
                    </div>
                </CollapsibleSection>
            )}

            {/* Improvement areas */}
            {improvement_areas.length > 0 && (
                <div className="rounded-xl border border-amber-500/15 bg-amber-500/5 p-4">
                    <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        <p className="text-sm font-semibold text-amber-400">Priority Focus Areas</p>
                    </div>
                    <ul className="space-y-1.5">
                        {improvement_areas.map((area, i) => (
                            <li key={i} className="text-xs text-white/50 flex items-start gap-2">
                                <span className="text-amber-500/50 mt-0.5">→</span> {area}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
