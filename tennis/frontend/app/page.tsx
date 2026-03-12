import Link from 'next/link';
import { Upload, LayoutDashboard, User, ArrowRight, Activity, Zap, Target, TrendingUp } from 'lucide-react';

export default function LandingPage() {
    return (
        <div className="relative overflow-hidden">
            {/* Background grid */}
            <div
                className="absolute inset-0 opacity-[0.03]"
                style={{
                    backgroundImage: 'linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)',
                    backgroundSize: '48px 48px',
                }}
            />

            {/* Hero */}
            <section className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-28 pb-24 text-center">
                <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-brand-600/10 border border-brand-500/20 text-brand-400 text-sm font-medium mb-8 animate-fade-in">
                    <Activity className="w-3.5 h-3.5" />
                    Automated match analysis
                </div>

                <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-white tracking-tight mb-6 animate-slide-up">
                    Understand every{' '}
                    <span className="text-gradient">point you play</span>
                </h1>

                <p className="text-xl text-white/50 max-w-2xl mx-auto mb-10 animate-slide-up">
                    Upload your match video. TennisIQ automatically detects ball position, player movement,
                    shot speed, and line calls — and delivers a complete match report.
                </p>

                <div className="flex flex-wrap items-center justify-center gap-4 animate-slide-up">
                    <Link href="/upload" className="btn-primary text-base px-7 py-3.5">
                        Upload a match video
                        <ArrowRight className="w-4 h-4" />
                    </Link>
                    <Link href="/dashboard" className="btn-secondary text-base px-7 py-3.5">
                        View dashboard
                    </Link>
                </div>
            </section>

            {/* Feature cards */}
            <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-24">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {FEATURES.map((f) => (
                        <div key={f.label} className="card-hover group">
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 ${f.iconBg}`}>
                                <f.icon className="w-5 h-5" />
                            </div>
                            <h3 className="text-sm font-semibold text-white mb-1">{f.label}</h3>
                            <p className="text-xs text-white/40 leading-relaxed">{f.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Main CTA tiles */}
            <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-32">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {CTA_CARDS.map((c) => (
                        <Link key={c.href} href={c.href} className="group card-hover flex flex-col gap-4">
                            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${c.iconBg}`}>
                                <c.icon className="w-6 h-6" />
                            </div>
                            <div>
                                <h2 className="text-lg font-bold text-white">{c.title}</h2>
                                <p className="text-sm text-white/40 mt-1">{c.subtitle}</p>
                            </div>
                            <div className="flex items-center text-brand-400 text-sm font-medium mt-auto group-hover:gap-2 gap-1 transition-all duration-200">
                                {c.cta} <ArrowRight className="w-4 h-4" />
                            </div>
                        </Link>
                    ))}
                </div>
            </section>
        </div>
    );
}

const FEATURES = [
    { label: 'Ball tracking', description: 'Frame-by-frame ball position and trajectory analysis.', icon: Target, iconBg: 'bg-court-500/20 text-court-400' },
    { label: 'Shot speed', description: 'Automatic speed measurement for every shot.', icon: Zap, iconBg: 'bg-amber-500/20 text-amber-400' },
    { label: 'Line calls', description: 'Computer vision line call decisions on every bounce.', icon: Activity, iconBg: 'bg-brand-500/20 text-brand-400' },
    { label: 'Trend analysis', description: 'Track your performance improvements over time.', icon: TrendingUp, iconBg: 'bg-red-500/20 text-red-400' },
];

const CTA_CARDS = [
    {
        href: '/upload',
        title: 'Upload match video',
        subtitle: 'MP4, MOV, or MKV. Analysis starts automatically.',
        cta: 'Start upload',
        icon: Upload,
        iconBg: 'bg-brand-600/20 text-brand-400',
    },
    {
        href: '/dashboard',
        title: 'View past matches',
        subtitle: 'Browse your analytics history and compare sessions.',
        cta: 'Open dashboard',
        icon: LayoutDashboard,
        iconBg: 'bg-court-500/20 text-court-400',
    },
    {
        href: '/register',
        title: 'Create account',
        subtitle: 'Free plan includes 3 uploads per month.',
        cta: 'Get started',
        icon: User,
        iconBg: 'bg-amber-500/20 text-amber-400',
    },
];
