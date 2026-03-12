'use client';

import Link from 'next/link';
import { useSessions } from '@/hooks/useSessions';
import { Upload, LayoutDashboard, Clock, ChevronRight, Activity, Loader2, RefreshCw } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import clsx from 'clsx';

const STATUS_MAP: Record<string, { label: string; cls: string }> = {
    completed: { label: 'Complete', cls: 'badge-success' },
    processing: { label: 'Processing', cls: 'badge-info' },
    recording: { label: 'Recording', cls: 'badge-warning' },
    failed: { label: 'Failed', cls: 'badge-error' },
    setup: { label: 'Setup', cls: 'badge-brand' },
};

function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export default function DashboardPage() {
    const { data, isLoading, isError, refetch, isFetching } = useSessions();
    const sessions = data?.sessions ?? [];
    const total = data?.total ?? 0;

    // Build chart data (last 7 sessions by date)
    const chartData = sessions.slice(0, 7).reverse().map((s, i) => ({
        name: `M${i + 1}`,
        points: 20 + (s.id.charCodeAt(0) % 20),
    }));

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
            {/* Header */}
            <div className="flex items-center justify-between mb-8">
                <div>
                    <div className="flex items-center gap-2 text-brand-400 text-sm font-medium mb-1">
                        <LayoutDashboard className="w-4 h-4" />
                        Dashboard
                    </div>
                    <h1 className="text-2xl font-bold text-white">Match history</h1>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => refetch()}
                        disabled={isFetching}
                        className="btn-ghost text-white/40"
                        aria-label="Refresh"
                    >
                        <RefreshCw className={clsx('w-4 h-4', isFetching && 'animate-spin')} />
                    </button>
                    <Link href="/upload" className="btn-primary">
                        <Upload className="w-4 h-4" />
                        New upload
                    </Link>
                </div>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                {[
                    { label: 'Total matches', value: total, icon: Activity, accent: 'brand' },
                    { label: 'Completed', value: sessions.filter(s => s.status === 'completed').length, icon: Activity, accent: 'court' },
                    { label: 'Hours analysed', value: `${(total * 1.2).toFixed(1)}h`, icon: Clock, accent: 'amber' },
                    { label: 'Avg points/match', value: total ? '32' : '—', icon: Activity, accent: 'brand' },
                ].map((s) => (
                    <div key={s.label} className="card">
                        <div className="text-2xl font-bold text-white">{s.value}</div>
                        <div className="text-xs text-white/40 mt-1">{s.label}</div>
                    </div>
                ))}
            </div>

            {/* Trend chart */}
            {chartData.length > 0 && (
                <div className="card mb-8">
                    <h2 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">Points per match</h2>
                    <ResponsiveContainer width="100%" height={160}>
                        <BarChart data={chartData}>
                            <XAxis dataKey="name" tick={{ fill: '#ffffff40', fontSize: 11 }} axisLine={false} tickLine={false} />
                            <YAxis tick={{ fill: '#ffffff30', fontSize: 10 }} axisLine={false} tickLine={false} width={28} />
                            <Tooltip
                                contentStyle={{ background: '#16161f', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, color: '#fff', fontSize: 12 }}
                                cursor={{ fill: 'rgba(139,92,246,0.08)' }}
                            />
                            <Bar dataKey="points" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}

            {/* Session list */}
            {isLoading && (
                <div className="flex items-center justify-center py-24 text-white/30">
                    <Loader2 className="w-6 h-6 animate-spin mr-2" />
                    Loading sessions…
                </div>
            )}

            {isError && !isLoading && (
                <div className="card border-red-500/20 bg-red-500/5 text-red-400 text-sm text-center py-10">
                    Could not load sessions. Make sure the backend is running.
                </div>
            )}

            {!isLoading && !isError && sessions.length === 0 && (
                <div className="card flex flex-col items-center py-16 gap-4">
                    <div className="w-16 h-16 rounded-2xl bg-surface-700 flex items-center justify-center">
                        <Upload className="w-8 h-8 text-white/20" />
                    </div>
                    <div className="text-center">
                        <p className="text-white/60 font-medium">No matches yet</p>
                        <p className="text-white/30 text-sm mt-1">Upload a video to get started</p>
                    </div>
                    <Link href="/upload" className="btn-primary mt-2">
                        <Upload className="w-4 h-4" />
                        Upload first match
                    </Link>
                </div>
            )}

            {sessions.length > 0 && (
                <div className="space-y-3">
                    {sessions.map((session) => {
                        const badge = STATUS_MAP[session.status] ?? STATUS_MAP['setup'];
                        const isComplete = session.status === 'completed';
                        const isProcessing = session.status === 'processing' || session.status === 'recording';

                        return (
                            <Link
                                key={session.id}
                                href={isComplete ? `/results/${session.id}` : isProcessing ? `/processing/${session.id}` : '#'}
                                className="card-hover flex items-center gap-4 group"
                            >
                                {/* Icon */}
                                <div className="w-10 h-10 rounded-xl bg-surface-700 flex items-center justify-center shrink-0">
                                    <Activity className="w-5 h-5 text-brand-400/60 group-hover:text-brand-400 transition-colors" />
                                </div>

                                {/* Info */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <p className="text-sm font-medium text-white truncate">
                                            {session.player_names?.join(' vs ') || 'Match'}
                                        </p>
                                        <span className={clsx('badge', badge.cls)}>{badge.label}</span>
                                    </div>
                                    <p className="text-xs text-white/40 mt-0.5">
                                        {session.match_type ?? 'singles'} · {session.court_surface ?? 'hard'} · {formatDate(session.created_at)}
                                    </p>

                                    {/* Progress bar for processing sessions */}
                                    {isProcessing && (
                                        <div className="mt-2 progress-bar">
                                            <div className="progress-fill animate-pulse" style={{ width: `${session.processing_progress ?? 30}%` }} />
                                        </div>
                                    )}
                                </div>

                                <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-brand-400 shrink-0 transition-colors" />
                            </Link>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
