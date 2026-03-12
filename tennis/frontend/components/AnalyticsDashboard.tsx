'use client';

import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Area, AreaChart } from 'recharts';
import { Activity, TrendingUp, Target, Zap, Brain, ChevronRight, AlertTriangle, CheckCircle } from 'lucide-react';
import clsx from 'clsx';

// ─── Types ──────────────────────────────────────────

interface ShotData {
    name: string;
    count: number;
    percentage: number;
    wins: number;
    successRate: number;
}

interface RallyData {
    totalRallies: number;
    avgLength: number;
    maxLength: number;
    distribution: Record<string, number>;
    shortPct: number;
    mediumPct: number;
    longPct: number;
    winRateByLength: Record<string, number>;
}

interface MomentumPoint {
    point: number;
    winPct: number;
}

interface MomentumData {
    rollingWinPct: MomentumPoint[];
    momentumShifts: { point: number; direction: string; magnitude: number }[];
    consistencyScore: number;
}

interface FatiguePhase {
    phase: string;
    points: number;
    winRate: number;
    unforcedErrors: number;
    avgRallyLength: number;
    avgSpeedMph: number;
}

interface ImprovementArea {
    area: string;
    metric: string;
    recommendation: string;
    priority: number;
}

// ─── Color Palette ──────────────────────────────────

const COLORS = ['#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899', '#14b8a6', '#a78bfa'];

const PHASE_COLORS: Record<string, string> = {
    early: '#22c55e',
    middle: '#f59e0b',
    late: '#ef4444',
};

// ─── Tabs ───────────────────────────────────────────

type TabId = 'shots' | 'rallies' | 'momentum' | 'coaching';

const TABS: { id: TabId; label: string; icon: typeof Activity }[] = [
    { id: 'shots', label: 'Shot Analysis', icon: Target },
    { id: 'rallies', label: 'Rally Analytics', icon: Activity },
    { id: 'momentum', label: 'Momentum', icon: TrendingUp },
    { id: 'coaching', label: 'Coaching', icon: Brain },
];

// ─── Main Component ─────────────────────────────────

interface AnalyticsDashboardProps {
    matchId?: string;
    playerId?: string;
}

export function AnalyticsDashboard({ matchId, playerId }: AnalyticsDashboardProps) {
    const [activeTab, setActiveTab] = useState<TabId>('shots');

    // Demo data — in production, fetched from API via react-query
    const shotData: ShotData[] = [
        { name: 'Forehand', count: 45, percentage: 35.2, wins: 12, successRate: 26.7 },
        { name: 'Backhand', count: 32, percentage: 25.0, wins: 8, successRate: 25.0 },
        { name: 'Serve', count: 28, percentage: 21.9, wins: 6, successRate: 21.4 },
        { name: 'Volley', count: 12, percentage: 9.4, wins: 5, successRate: 41.7 },
        { name: 'Slice', count: 8, percentage: 6.3, wins: 2, successRate: 25.0 },
        { name: 'Drop Shot', count: 3, percentage: 2.3, wins: 1, successRate: 33.3 },
    ];

    const rallyData: RallyData = {
        totalRallies: 48, avgLength: 6.2, maxLength: 22,
        distribution: { '0-2': 8, '3-5': 15, '6-8': 12, '9-11': 7, '12-14': 4, '15+': 2 },
        shortPct: 16.7, mediumPct: 56.3, longPct: 27.1,
        winRateByLength: { short: 62.5, medium: 53.3, long: 38.5 },
    };

    const momentumData: MomentumData = {
        rollingWinPct: Array.from({ length: 40 }, (_, i) => ({
            point: i + 1,
            winPct: 50 + Math.sin(i / 5) * 25 + (Math.random() - 0.5) * 10,
        })),
        momentumShifts: [
            { point: 12, direction: 'positive', magnitude: 25.3 },
            { point: 28, direction: 'negative', magnitude: 22.1 },
        ],
        consistencyScore: 0.72,
    };

    const fatigueData: FatiguePhase[] = [
        { phase: 'early', points: 16, winRate: 68.8, unforcedErrors: 2, avgRallyLength: 5.1, avgSpeedMph: 78.3 },
        { phase: 'middle', points: 16, winRate: 56.3, unforcedErrors: 4, avgRallyLength: 6.8, avgSpeedMph: 74.1 },
        { phase: 'late', points: 16, winRate: 43.8, unforcedErrors: 6, avgRallyLength: 7.2, avgSpeedMph: 69.5 },
    ];

    const improvements: ImprovementArea[] = [
        { area: 'Backhand consistency', metric: '25% success rate (32 attempts)', recommendation: 'Focus on backhand crosscourt placement — practice with targets', priority: 0.8 },
        { area: 'Late-match endurance', metric: '43.8% win rate in final third', recommendation: 'Increase fitness training — interval sprints and recovery drills', priority: 0.7 },
        { area: 'Net approach timing', metric: 'Only 9.4% net approach rate', recommendation: 'Look for short balls to approach on — practice split step at service line', priority: 0.5 },
    ];

    return (
        <div className="space-y-6">
            {/* Tab Navigation */}
            <div className="flex gap-1 p-1 bg-surface-800 rounded-xl border border-white/[0.06]">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={clsx(
                            'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 flex-1 justify-center',
                            activeTab === tab.id
                                ? 'bg-brand-600 text-white shadow-lg shadow-brand-900/40'
                                : 'text-white/50 hover:text-white hover:bg-white/5'
                        )}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {activeTab === 'shots' && <ShotAnalyticsPanel data={shotData} />}
            {activeTab === 'rallies' && <RallyAnalyticsPanel data={rallyData} />}
            {activeTab === 'momentum' && <MomentumPanel data={momentumData} fatigue={fatigueData} />}
            {activeTab === 'coaching' && <CoachingPanel improvements={improvements} />}
        </div>
    );
}

// ─── Shot Analytics Panel ───────────────────────────

function ShotAnalyticsPanel({ data }: { data: ShotData[] }) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Shot Distribution Chart */}
            <div className="card">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4">Shot Distribution</h3>
                <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis type="number" stroke="#ffffff40" tick={{ fill: '#ffffff60', fontSize: 11 }} />
                        <YAxis type="category" dataKey="name" stroke="#ffffff40" tick={{ fill: '#ffffff80', fontSize: 12 }} width={80} />
                        <Tooltip
                            contentStyle={{ background: '#1a1a24', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }}
                            formatter={(value: number, name: string) => [value, name === 'count' ? 'Count' : 'Wins']}
                        />
                        <Bar dataKey="count" fill="#8b5cf6" radius={[0, 6, 6, 0]} barSize={20} />
                        <Bar dataKey="wins" fill="#22c55e" radius={[0, 6, 6, 0]} barSize={20} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Shot Type Pie Chart */}
            <div className="card">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4">Shot Mix</h3>
                <ResponsiveContainer width="100%" height={280}>
                    <PieChart>
                        <Pie data={data} dataKey="percentage" nameKey="name" cx="50%" cy="50%" innerRadius={60} outerRadius={100} strokeWidth={2} stroke="#111118">
                            {data.map((_, i) => (
                                <Cell key={i} fill={COLORS[i % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip contentStyle={{ background: '#1a1a24', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }} />
                    </PieChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-3 gap-2 mt-2">
                    {data.slice(0, 6).map((s, i) => (
                        <div key={s.name} className="flex items-center gap-2 text-xs text-white/60">
                            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                            {s.name}
                        </div>
                    ))}
                </div>
            </div>

            {/* Success Rate Table */}
            <div className="card lg:col-span-2">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4">Shot Effectiveness</h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="border-b border-white/[0.06]">
                                <th className="text-left py-3 px-2 text-white/50 font-medium">Shot Type</th>
                                <th className="text-right py-3 px-2 text-white/50 font-medium">Count</th>
                                <th className="text-right py-3 px-2 text-white/50 font-medium">% of Total</th>
                                <th className="text-right py-3 px-2 text-white/50 font-medium">Winners</th>
                                <th className="text-right py-3 px-2 text-white/50 font-medium">Success Rate</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((s, i) => (
                                <tr key={s.name} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                                    <td className="py-3 px-2 flex items-center gap-2">
                                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                                        <span className="text-white font-medium">{s.name}</span>
                                    </td>
                                    <td className="text-right py-3 px-2 text-white/80">{s.count}</td>
                                    <td className="text-right py-3 px-2 text-white/60">{s.percentage}%</td>
                                    <td className="text-right py-3 px-2 text-court-400 font-medium">{s.wins}</td>
                                    <td className="text-right py-3 px-2">
                                        <span className={clsx(
                                            'px-2 py-0.5 rounded-full text-xs font-semibold',
                                            s.successRate >= 30 ? 'bg-court-500/20 text-court-400' : 'bg-amber-500/20 text-amber-400'
                                        )}>
                                            {s.successRate}%
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

// ─── Rally Analytics Panel ──────────────────────────

function RallyAnalyticsPanel({ data }: { data: RallyData }) {
    const distData = Object.entries(data.distribution).map(([range, count]) => ({ range, count }));
    const winRateData = Object.entries(data.winRateByLength).map(([type, rate]) => ({ type: type.charAt(0).toUpperCase() + type.slice(1), rate }));

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Rally Stats */}
            <div className="card flex flex-col gap-4">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider">Rally Summary</h3>
                <div className="space-y-4">
                    <div>
                        <div className="stat-number text-brand-400">{data.avgLength}</div>
                        <div className="stat-label">Avg Rally Length</div>
                    </div>
                    <div className="divider" />
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <div className="text-xl font-bold text-white">{data.totalRallies}</div>
                            <div className="text-xs text-white/40">Total Rallies</div>
                        </div>
                        <div>
                            <div className="text-xl font-bold text-court-400">{data.maxLength}</div>
                            <div className="text-xs text-white/40">Longest Rally</div>
                        </div>
                    </div>
                    <div className="divider" />
                    <div className="space-y-2">
                        {[
                            { label: 'Short (1-4)', pct: data.shortPct, color: '#22c55e' },
                            { label: 'Medium (5-9)', pct: data.mediumPct, color: '#f59e0b' },
                            { label: 'Long (10+)', pct: data.longPct, color: '#8b5cf6' },
                        ].map((b) => (
                            <div key={b.label}>
                                <div className="flex justify-between text-xs mb-1">
                                    <span className="text-white/60">{b.label}</span>
                                    <span className="text-white/80 font-medium">{b.pct}%</span>
                                </div>
                                <div className="progress-bar">
                                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${b.pct}%`, backgroundColor: b.color }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Rally Length Distribution */}
            <div className="card">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4">Length Distribution</h3>
                <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={distData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="range" stroke="#ffffff40" tick={{ fill: '#ffffff60', fontSize: 11 }} />
                        <YAxis stroke="#ffffff40" tick={{ fill: '#ffffff60', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#1a1a24', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }} />
                        <Bar dataKey="count" fill="#8b5cf6" radius={[6, 6, 0, 0]} barSize={32} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Win Rate by Rally Length */}
            <div className="card">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4">Win Rate by Length</h3>
                <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={winRateData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="type" stroke="#ffffff40" tick={{ fill: '#ffffff60', fontSize: 11 }} />
                        <YAxis domain={[0, 100]} stroke="#ffffff40" tick={{ fill: '#ffffff60', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#1a1a24', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }} formatter={(v: number) => [`${v}%`, 'Win Rate']} />
                        <Bar dataKey="rate" radius={[6, 6, 0, 0]} barSize={40}>
                            {winRateData.map((_, i) => (
                                <Cell key={i} fill={COLORS[i % COLORS.length]} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

// ─── Momentum Panel ─────────────────────────────────

function MomentumPanel({ data, fatigue }: { data: MomentumData; fatigue: FatiguePhase[] }) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Momentum Chart */}
            <div className="card lg:col-span-2">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider">Match Momentum</h3>
                    <div className="flex items-center gap-2">
                        <Zap className="w-3.5 h-3.5 text-amber-400" />
                        <span className="text-xs text-white/60">Consistency: <span className="text-white font-medium">{(data.consistencyScore * 100).toFixed(0)}%</span></span>
                    </div>
                </div>
                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={data.rollingWinPct}>
                        <defs>
                            <linearGradient id="momentumGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis dataKey="point" stroke="#ffffff40" tick={{ fill: '#ffffff60', fontSize: 11 }} label={{ value: 'Point', position: 'insideBottom', offset: -5, fill: '#ffffff40', fontSize: 11 }} />
                        <YAxis domain={[0, 100]} stroke="#ffffff40" tick={{ fill: '#ffffff60', fontSize: 11 }} label={{ value: 'Win %', angle: -90, position: 'insideLeft', fill: '#ffffff40', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#1a1a24', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', color: '#fff' }} formatter={(v: number) => [`${v.toFixed(1)}%`, 'Win Rate']} />
                        <Area type="monotone" dataKey="winPct" stroke="#8b5cf6" fill="url(#momentumGrad)" strokeWidth={2} dot={false} />
                        {/* 50% line */}
                        <Line type="monotone" dataKey={() => 50} stroke="#ffffff20" strokeDasharray="5 5" dot={false} />
                    </AreaChart>
                </ResponsiveContainer>
                {data.momentumShifts.length > 0 && (
                    <div className="mt-3 flex gap-2 flex-wrap">
                        {data.momentumShifts.map((s, i) => (
                            <span key={i} className={clsx('badge', s.direction === 'positive' ? 'badge-success' : 'badge-error')}>
                                Point {s.point}: {s.direction === 'positive' ? '↑' : '↓'} {s.magnitude.toFixed(0)}%
                            </span>
                        ))}
                    </div>
                )}
            </div>

            {/* Fatigue Analysis */}
            <div className="card">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4">Fatigue Analysis</h3>
                <div className="space-y-4">
                    {fatigue.map((p) => (
                        <div key={p.phase} className="p-3 rounded-xl bg-surface-700/50 border border-white/[0.04]">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: PHASE_COLORS[p.phase] }}>
                                    {p.phase} Phase
                                </span>
                                <span className="text-lg font-bold text-white">{p.winRate}%</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-xs text-white/50">
                                <div>UE: <span className="text-white/80">{p.unforcedErrors}</span></div>
                                <div>Rally: <span className="text-white/80">{p.avgRallyLength}</span></div>
                                <div>Speed: <span className="text-white/80">{p.avgSpeedMph} mph</span></div>
                                <div>Points: <span className="text-white/80">{p.points}</span></div>
                            </div>
                            <div className="mt-2 progress-bar">
                                <div className="h-full rounded-full" style={{ width: `${p.winRate}%`, backgroundColor: PHASE_COLORS[p.phase] }} />
                            </div>
                        </div>
                    ))}
                    {fatigue.length >= 2 && fatigue[0].winRate > fatigue[fatigue.length - 1].winRate * 1.15 && (
                        <div className="flex items-start gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/20">
                            <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-red-300">Win rate declined from {fatigue[0].winRate}% to {fatigue[fatigue.length - 1].winRate}% — potential fatigue detected</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// ─── Coaching Insights Panel ────────────────────────

function CoachingPanel({ improvements }: { improvements: ImprovementArea[] }) {
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Improvement Areas */}
            <div className="card lg:col-span-2">
                <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4">Areas for Improvement</h3>
                <div className="space-y-3">
                    {improvements.map((item, i) => (
                        <div key={i} className="p-4 rounded-xl bg-surface-700/50 border border-white/[0.04] hover:border-brand-600/30 transition-all duration-200">
                            <div className="flex items-start gap-3">
                                <div className={clsx(
                                    'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5',
                                    item.priority >= 0.7 ? 'bg-amber-500/20 text-amber-400' : 'bg-blue-500/20 text-blue-400'
                                )}>
                                    {item.priority >= 0.7 ? <AlertTriangle className="w-4 h-4" /> : <Target className="w-4 h-4" />}
                                </div>
                                <div className="flex-1">
                                    <div className="text-sm font-semibold text-white mb-1">{item.area}</div>
                                    <div className="text-xs text-white/50 mb-2">{item.metric}</div>
                                    <div className="flex items-start gap-1.5">
                                        <CheckCircle className="w-3.5 h-3.5 text-court-400 mt-0.5 flex-shrink-0" />
                                        <span className="text-xs text-court-300">{item.recommendation}</span>
                                    </div>
                                </div>
                                <ChevronRight className="w-4 h-4 text-white/20 mt-1" />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
