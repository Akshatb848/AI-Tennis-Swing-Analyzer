'use client';

import { useState, useCallback } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { VideoPlayer, type FrameOverlay } from '@/components/VideoPlayer';
import { FrameTimeline } from '@/components/FrameTimeline';
import { CoachingPanel } from '@/components/CoachingPanel';
import { CoachingReport } from '@/components/CoachingReport';
import { StatCard } from '@/components/StatCard';
import { RallyTimeline, type RallyEntry } from '@/components/RallyTimeline';
import { useVoiceCoach } from '@/hooks/useVoiceCoach';
import {
    Activity, Target, Zap, BarChart2, User2, BookOpen,
    ArrowLeft, Download, Loader2, AlertCircle,
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts';
import clsx from 'clsx';

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ─── Tab definitions ─────────────────────────────────────────────────────────

type TabId = 'coaching' | 'summary' | 'players' | 'rallies';

const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
    { id: 'coaching', label: 'Coaching Report', icon: BookOpen },
    { id: 'summary', label: 'Match Stats', icon: BarChart2 },
    { id: 'players', label: 'Per Player', icon: User2 },
    { id: 'rallies', label: 'Rallies', icon: Activity },
];

// ─── Fetchers ─────────────────────────────────────────────────────────────────

async function fetchResults(jobId: string) {
    const { data } = await axios.get(`${BACKEND}/api/v1/analyze/${jobId}/results`);
    return data;
}
async function fetchOverlays(jobId: string) {
    const { data } = await axios.get(`${BACKEND}/api/v1/analyze/${jobId}/overlays`);
    return data;
}
async function fetchVoiceScript(jobId: string) {
    const { data } = await axios.get(`${BACKEND}/api/v1/analyze/${jobId}/voice-script`);
    return data;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function shotColor(type: string) {
    const m: Record<string, string> = {
        forehand: '#8b5cf6', backhand: '#3b82f6', serve: '#f59e0b',
        volley: '#10b981', smash: '#ef4444', slice: '#06b6d4',
    };
    return m[type] ?? '#6b7280';
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export default function ResultsPage() {
    const params = useParams();
    const router = useRouter();
    const jobId = params?.id as string;

    const [tab, setTab] = useState<TabId>('coaching');
    const [currentTime, setCurrentTime] = useState(0);
    const [jumpTo, setJumpTo] = useState<number | null>(null);
    const [activeOverlay, setActiveOverlay] = useState<FrameOverlay | null>(null);
    const [voiceEnabled, setVoiceEnabled] = useState(false);

    // Main analysis results
    const { data: results, isLoading, error } = useQuery({
        queryKey: ['results', jobId],
        queryFn: () => fetchResults(jobId),
        enabled: !!jobId,
        retry: 2,
    });

    // Frame overlays from MoE pipeline
    const { data: overlaysData } = useQuery({
        queryKey: ['overlays', jobId],
        queryFn: () => fetchOverlays(jobId),
        enabled: !!jobId && !!results,
        retry: 1,
    });

    // Voice script
    const { data: voiceData } = useQuery({
        queryKey: ['voice', jobId],
        queryFn: () => fetchVoiceScript(jobId),
        enabled: !!jobId && !!results,
        retry: 1,
    });

    // Voice coach hook
    const { speak, stop, isSpeaking } = useVoiceCoach({
        script: voiceData?.script ?? [],
        currentTime,
        enabled: voiceEnabled,
    });

    const handleTimeUpdate = useCallback((t: number) => setCurrentTime(t), []);
    const handleSeek = useCallback((t: number) => setJumpTo(t), []);
    const handleOverlayChange = useCallback((ov: FrameOverlay | null) => setActiveOverlay(ov), []);

    // Derived data
    const stats = results?.results?.stats ?? {};
    const shots = results?.results?.shots ?? {};
    const rallies = results?.results?.rallies ?? [];
    const perPlayer = results?.results?.per_player ?? {};
    const coachingReport = results?.results?.coaching_report ?? null;
    const uploadId = results?.upload_id ?? '';
    const videoDuration = results?.video?.duration_seconds ?? 0;

    // Video stream URL — served directly from backend
    const videoSrc = uploadId
        ? `${BACKEND}/api/v1/upload/${uploadId}/stream`
        : '';

    // Frame overlays
    const frameOverlays: FrameOverlay[] = overlaysData?.overlays ?? [];

    // Shot distribution for bar chart
    const shotDist = Object.entries(shots.distribution ?? {}).map(([k, v]) => ({
        name: k.charAt(0).toUpperCase() + k.slice(1),
        count: v as number,
        color: shotColor(k),
    }));

    // Rating color
    const RATING_COLORS: Record<string, string> = {
        Elite: '#10b981', Advanced: '#8b5cf6',
        Intermediate: '#f59e0b', Beginner: '#ef4444',
    };
    const ratingColor = RATING_COLORS[coachingReport?.overall_rating ?? ''] ?? '#8b5cf6';

    // Shot quality radar data
    const shotScores = coachingReport?.sections?.shot_quality?.scores ?? {};
    const radarData = Object.entries(shotScores).map(([k, v]) => ({
        subject: k.charAt(0).toUpperCase() + k.slice(1),
        score: (v as number) * 10,
        fullMark: 100,
    }));

    // ── Loading / Error states ────────────────────────────────────────────────

    if (isLoading) {
        return (
            <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4">
                <Loader2 className="w-8 h-8 animate-spin text-violet-400" />
                <p className="text-white/40">Loading analysis results…</p>
            </div>
        );
    }

    if (error || !results) {
        return (
            <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4 px-4 text-center">
                <AlertCircle className="w-8 h-8 text-red-400" />
                <p className="text-white font-semibold">Analysis not found</p>
                <p className="text-white/40 text-sm">This session may have expired. Please upload and analyze again.</p>
                <button onClick={() => router.push('/upload')} className="btn-primary mt-2">
                    Upload video
                </button>
            </div>
        );
    }

    // ── Main render ────────────────────────────────────────────────────────────

    return (
        <div className="max-w-[1400px] mx-auto px-4 py-6 space-y-4">

            {/* Breadcrumb */}
            <div className="flex items-center justify-between">
                <button
                    onClick={() => router.back()}
                    className="flex items-center gap-1.5 text-sm text-white/40 hover:text-white transition-colors"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back
                </button>
                <div className="flex items-center gap-2">
                    {coachingReport?.overall_rating && (
                        <span
                            className="text-xs font-bold px-2.5 py-1 rounded-full"
                            style={{ background: `${ratingColor}20`, color: ratingColor }}
                        >
                            {coachingReport.overall_rating} · {coachingReport.overall_score?.toFixed(1)}/10
                        </span>
                    )}
                    <button className="flex items-center gap-1.5 text-xs text-white/40 hover:text-white/70 transition-colors border border-white/10 rounded-lg px-2.5 py-1">
                        <Download className="w-3 h-3" />
                        Export
                    </button>
                </div>
            </div>

            {/* ── TOP ROW: Video + Coaching Panel ──────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">

                {/* Video Player */}
                <div className="space-y-3">
                    {videoSrc ? (
                        <VideoPlayer
                            src={videoSrc}
                            frameOverlays={frameOverlays}
                            onTimeUpdate={handleTimeUpdate}
                            onActiveOverlayChange={handleOverlayChange}
                            jumpTo={jumpTo}
                            className="w-full aspect-video"
                        />
                    ) : (
                        <div className="w-full aspect-video rounded-2xl bg-white/3 border border-white/5 flex items-center justify-center">
                            <p className="text-white/30 text-sm">Video not available</p>
                        </div>
                    )}

                    {/* Frame Timeline */}
                    <div className="rounded-xl border border-white/5 bg-white/2 p-3">
                        <p className="text-xs text-white/30 uppercase tracking-wider mb-2.5 font-semibold">
                            Coaching moments
                        </p>
                        <FrameTimeline
                            overlays={frameOverlays}
                            duration={videoDuration}
                            currentTime={currentTime}
                            onSeek={handleSeek}
                        />
                    </div>
                </div>

                {/* Coaching Panel */}
                <div className="rounded-2xl border border-white/5 bg-white/2 p-4">
                    <CoachingPanel
                        activeOverlay={activeOverlay}
                        voiceEnabled={voiceEnabled}
                        onToggleVoice={() => {
                            if (voiceEnabled) stop();
                            setVoiceEnabled(!voiceEnabled);
                        }}
                        onSpeakText={speak}
                        overallRating={coachingReport?.overall_rating}
                        overallScore={coachingReport?.overall_score}
                        topIssue={coachingReport?.improvement_areas?.[0]}
                    />
                </div>
            </div>

            {/* ── STATS ROW ────────────────────────────────────────────────── */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <StatCard
                    label="Ball Speed"
                    value={`${stats.avg_shot_speed_kmh ?? '—'}`}
                    unit="km/h avg"
                    icon={<Zap className="w-4 h-4" />}
                    trend={stats.avg_shot_speed_kmh > 130 ? 'up' : 'neutral'}
                />
                <StatCard
                    label="Rallies"
                    value={`${stats.total_rallies ?? '—'}`}
                    unit="detected"
                    icon={<Activity className="w-4 h-4" />}
                    trend="neutral"
                />
                <StatCard
                    label="Shots"
                    value={`${stats.total_shots ?? '—'}`}
                    unit="classified"
                    icon={<Target className="w-4 h-4" />}
                    trend="neutral"
                />
                <StatCard
                    label="Technique"
                    value={`${stats.technique_score ?? coachingReport?.sections?.biomechanics?.score ?? '—'}`}
                    unit="/ 10"
                    icon={<BarChart2 className="w-4 h-4" />}
                    trend={(stats.technique_score ?? 0) >= 7 ? 'up' : 'neutral'}
                />
            </div>

            {/* ── TABS + CONTENT ────────────────────────────────────────────── */}
            <div className="rounded-2xl border border-white/5 bg-white/2 overflow-hidden">
                {/* Tab header */}
                <div className="flex border-b border-white/5 overflow-x-auto">
                    {TABS.map(t => {
                        const Icon = t.icon;
                        return (
                            <button
                                key={t.id}
                                onClick={() => setTab(t.id)}
                                className={clsx(
                                    'flex items-center gap-1.5 px-4 py-3 text-sm font-medium transition-all whitespace-nowrap',
                                    'border-b-2',
                                    tab === t.id
                                        ? 'border-violet-500 text-white bg-violet-500/5'
                                        : 'border-transparent text-white/40 hover:text-white/70',
                                )}
                            >
                                <Icon className="w-4 h-4" />
                                {t.label}
                            </button>
                        );
                    })}
                </div>

                <div className="p-5">

                    {/* ── Coaching Report tab ─────────────────────────────── */}
                    {tab === 'coaching' && (
                        coachingReport
                            ? <CoachingReport report={coachingReport} />
                            : (
                                <div className="text-center py-10 text-white/30">
                                    <BookOpen className="w-8 h-8 mx-auto mb-3 opacity-30" />
                                    <p>Coaching report not available for this session.</p>
                                </div>
                            )
                    )}

                    {/* ── Summary tab ─────────────────────────────────────── */}
                    {tab === 'summary' && (
                        <div className="space-y-6">
                            {/* Shot distribution bar chart */}
                            {shotDist.length > 0 && (
                                <div>
                                    <h3 className="text-sm font-semibold text-white/60 mb-3 uppercase tracking-wider">Shot Distribution</h3>
                                    <div className="h-52">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={shotDist} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
                                                <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} axisLine={false} tickLine={false} />
                                                <YAxis tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} axisLine={false} tickLine={false} />
                                                <Tooltip
                                                    contentStyle={{ background: '#0a0a0f', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                                                    labelStyle={{ color: 'white' }}
                                                    itemStyle={{ color: 'rgba(255,255,255,0.6)' }}
                                                />
                                                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                                    {shotDist.map((d, i) => <Cell key={i} fill={d.color} />)}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            )}

                            {/* Shot quality radar chart */}
                            {radarData.length > 0 && (
                                <div>
                                    <h3 className="text-sm font-semibold text-white/60 mb-3 uppercase tracking-wider">Shot Quality Radar</h3>
                                    <div className="h-56 flex items-center justify-center">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <RadarChart data={radarData}>
                                                <PolarGrid stroke="rgba(255,255,255,0.08)" />
                                                <PolarAngleAxis dataKey="subject" tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} />
                                                <Radar
                                                    name="Quality"
                                                    dataKey="score"
                                                    stroke="#8b5cf6"
                                                    fill="#8b5cf6"
                                                    fillOpacity={0.25}
                                                    strokeWidth={2}
                                                />
                                            </RadarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            )}

                            {/* Key stats grid */}
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                                {[
                                    { label: 'Winners', value: stats.winners },
                                    { label: 'Errors', value: stats.errors },
                                    { label: 'Total Points', value: stats.total_points },
                                    { label: 'Max Speed', value: stats.max_shot_speed_kmh ? `${stats.max_shot_speed_kmh} km/h` : '—' },
                                    { label: 'Longest Rally', value: stats.longest_rally ? `${stats.longest_rally} shots` : '—' },
                                    { label: 'Ball Detections', value: stats.ball_detections },
                                ].map(row => (
                                    <div key={row.label} className="bg-white/3 rounded-lg p-3 text-center">
                                        <p className="text-lg font-bold text-white">{row.value ?? '—'}</p>
                                        <p className="text-xs text-white/40 mt-0.5">{row.label}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* ── Per Player tab ──────────────────────────────────── */}
                    {tab === 'players' && (
                        <div className="space-y-4">
                            {Object.entries(perPlayer).length > 0
                                ? Object.entries(perPlayer).map(([name, pStats]: [string, any]) => (
                                    <div key={name} className="rounded-xl border border-white/5 p-4">
                                        <p className="text-sm font-semibold text-white mb-3">{name}</p>
                                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                                            {Object.entries(pStats as Record<string, number>).map(([k, v]) => (
                                                <div key={k} className="bg-white/3 rounded-lg p-2 text-center">
                                                    <p className="text-base font-bold text-white">{typeof v === 'number' ? v.toFixed(1) : v}</p>
                                                    <p className="text-xs text-white/40 capitalize">{k.replace(/_/g, ' ')}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))
                                : (
                                    <div className="text-center py-10 text-white/30">
                                        <User2 className="w-8 h-8 mx-auto mb-3 opacity-30" />
                                        <p>Per-player stats not available.</p>
                                    </div>
                                )
                            }
                        </div>
                    )}

                    {/* ── Rallies tab ─────────────────────────────────────── */}
                    {tab === 'rallies' && (
                        Array.isArray(rallies) && rallies.length > 0
                            ? <RallyTimeline rallies={rallies as RallyEntry[]} />
                            : (
                                <div className="text-center py-10 text-white/30">
                                    <Activity className="w-8 h-8 mx-auto mb-3 opacity-30" />
                                    <p>Rally data not available.</p>
                                </div>
                            )
                    )}

                </div>
            </div>
        </div>
    );
}
