'use client';

import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowRight, Loader2 } from 'lucide-react';
import clsx from 'clsx';

const MATCH_TYPES = [
    { value: 'singles', label: 'Singles', desc: 'Standard 1v1 match' },
    { value: 'doubles', label: 'Doubles', desc: '2v2 team match' },
    { value: 'practice_rally', label: 'Practice Rally', desc: 'Training session — shot & rally analysis' },
];
const ENVIRONMENTS = [{ value: 'outdoor', label: 'Outdoor' }, { value: 'indoor', label: 'Indoor' }];
const SURFACES = [{ value: 'hard', label: 'Hard' }, { value: 'clay', label: 'Clay' }, { value: 'grass', label: 'Grass' }];
const FORMATS = [
    { value: 'pro_set', label: '1 set (Pro set)' },
    { value: 'best_of_3', label: 'Best of 3 sets' },
    { value: 'best_of_5', label: 'Best of 5 sets' },
];

export default function ConfigurePage() {
    const router = useRouter();
    const params = useSearchParams();

    const uploadId = params.get('upload_id') ?? '';
    const filename = params.get('filename') ?? 'video.mp4';

    const [matchType, setMatchType] = useState('singles');
    const [environment, setEnvironment] = useState('outdoor');
    const [surface, setSurface] = useState('hard');
    const [format, setFormat] = useState('best_of_3');
    const [p1Name, setP1Name] = useState('');
    const [p2Name, setP2Name] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const isPractice = matchType === 'practice_rally';
    const isDoubles = matchType === 'doubles';

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setSubmitting(true);
        setError(null);

        try {
            const sessionId = `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

            const form = new FormData();
            form.append('session_id', sessionId);
            form.append('upload_id', uploadId);
            form.append('match_type', matchType);
            form.append('filename', filename);

            const backendUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
            const res = await fetch(`${backendUrl}/api/v1/analyze`, {
                method: 'POST',
                body: form,
            });

            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Analysis failed (${res.status}): ${text}`);
            }

            const data = await res.json();
            const jobId = data.job_id as string;

            router.push(`/processing/${jobId}?session_id=${sessionId}&match_type=${matchType}`);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : 'Failed to start analysis. Is the backend running?');
            setSubmitting(false);
        }
    };

    return (
        <div className="max-w-2xl mx-auto px-4 py-16">
            {/* Stepper */}
            <div className="flex items-center gap-3 mb-10">
                {['Upload', 'Configure', 'Processing', 'Results'].map((step, i) => (
                    <div key={step} className="flex items-center gap-2">
                        {i > 0 && <div className="w-8 h-px bg-white/10" />}
                        <div className={clsx('step-dot', i === 0 ? 'step-dot-complete' : i === 1 ? 'step-dot-active' : 'step-dot-pending')}>
                            {i + 1}
                        </div>
                        <span className={clsx('text-xs font-medium', i <= 1 ? 'text-white' : 'text-white/30')}>{step}</span>
                    </div>
                ))}
            </div>

            <h1 className="text-3xl font-bold text-white mb-2">Configure match</h1>
            <p className="text-white/50 mb-2">
                {isPractice
                    ? 'Practice mode — we\'ll analyze shots, ball trajectory, and rally sequences.'
                    : 'Set up your match details, then hit Start.'}
            </p>
            <p className="text-xs text-white/30 mb-10 font-mono truncate">{filename}</p>

            <form onSubmit={handleSubmit} className="space-y-6">
                {/* Match type */}
                <div>
                    <label className="label">Match type</label>
                    <div className="grid grid-cols-3 gap-2">
                        {MATCH_TYPES.map((m) => (
                            <button
                                key={m.value}
                                type="button"
                                onClick={() => setMatchType(m.value)}
                                className={clsx(
                                    'py-3 px-4 rounded-xl border text-sm font-medium transition-all duration-200 text-left',
                                    matchType === m.value
                                        ? 'bg-brand-600/20 border-brand-500/50 text-brand-300'
                                        : 'bg-surface-700 border-white/10 text-white/60 hover:border-white/20',
                                )}
                            >
                                <div>{m.label}</div>
                                <div className="text-[10px] mt-0.5 opacity-60 font-normal">{m.desc}</div>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Player / Team names — hidden for practice rally */}
                {!isPractice && (
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="label">{isDoubles ? 'Team 1' : 'Player 1'}</label>
                            <input
                                type="text"
                                className="input"
                                value={p1Name}
                                onChange={(e) => setP1Name(e.target.value)}
                                placeholder={isDoubles ? 'Team 1' : 'Player 1'}
                            />
                        </div>
                        <div>
                            <label className="label">{isDoubles ? 'Team 2' : 'Player 2'}</label>
                            <input
                                type="text"
                                className="input"
                                value={p2Name}
                                onChange={(e) => setP2Name(e.target.value)}
                                placeholder={isDoubles ? 'Team 2' : 'Player 2'}
                            />
                        </div>
                    </div>
                )}

                {/* Environment + Surface — always shown */}
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="label">Environment</label>
                        <select className="select" value={environment} onChange={(e) => setEnvironment(e.target.value)}>
                            {ENVIRONMENTS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="label">Court surface</label>
                        <select className="select" value={surface} onChange={(e) => setSurface(e.target.value)}>
                            {SURFACES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
                        </select>
                    </div>
                </div>

                {/* Match format — ONLY for singles and doubles, NOT practice rally */}
                {!isPractice && (
                    <div>
                        <label className="label">Match format</label>
                        <select className="select" value={format} onChange={(e) => setFormat(e.target.value)}>
                            {FORMATS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
                        </select>
                    </div>
                )}

                {/* Practice rally info callout */}
                {isPractice && (
                    <div className="card border-brand-500/10 bg-brand-600/5 text-white/60 text-sm">
                        <p className="font-medium text-brand-300 mb-1">Practice mode</p>
                        <p>No match format needed. We'll focus on detecting shots, ball trajectory, and rally sequences from your training video.</p>
                    </div>
                )}

                {error && <div className="card border-red-500/20 bg-red-500/5 text-red-400 text-sm">{error}</div>}

                <button type="submit" disabled={submitting} className="btn-primary w-full py-4 text-base">
                    {submitting
                        ? <><Loader2 className="w-4 h-4 animate-spin" /> Starting analysis…</>
                        : <>Start analysis <ArrowRight className="w-4 h-4" /></>}
                </button>
            </form>
        </div>
    );
}
