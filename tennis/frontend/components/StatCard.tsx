'use client';

import clsx from 'clsx';
import { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface StatCardProps {
    label: string;
    value: string | number;
    unit?: string;
    trend?: 'up' | 'down' | 'neutral';
    trendValue?: string;
    icon?: ReactNode;
    accent?: 'brand' | 'court' | 'amber' | 'red';
    size?: 'sm' | 'md';
}

const ACCENT_MAP = {
    brand: { glow: 'shadow-brand-900/30', iconBg: 'bg-brand-600/20 text-brand-400', value: 'text-white' },
    court: { glow: 'shadow-court-900/30', iconBg: 'bg-court-500/20 text-court-400', value: 'text-court-400' },
    amber: { glow: 'shadow-amber-900/30', iconBg: 'bg-amber-500/20 text-amber-400', value: 'text-amber-400' },
    red: { glow: 'shadow-red-900/30', iconBg: 'bg-red-500/20 text-red-400', value: 'text-red-400' },
};

export function StatCard({
    label, value, unit, trend, trendValue, icon, accent = 'brand', size = 'md',
}: StatCardProps) {
    const a = ACCENT_MAP[accent];
    const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
    const trendColor = trend === 'up' ? 'text-court-400' : trend === 'down' ? 'text-red-400' : 'text-white/40';

    return (
        <div className={clsx('card flex flex-col gap-3 hover:shadow-lg transition-all duration-300', a.glow)}>
            <div className="flex items-start justify-between">
                {icon && (
                    <div className={clsx('w-9 h-9 rounded-xl flex items-center justify-center', a.iconBg)}>
                        {icon}
                    </div>
                )}
                {trend && trendValue && (
                    <div className={clsx('flex items-center gap-1 text-xs font-medium', trendColor)}>
                        <TrendIcon className="w-3 h-3" />
                        {trendValue}
                    </div>
                )}
            </div>
            <div>
                <div className={clsx('font-bold tracking-tight', size === 'md' ? 'text-3xl' : 'text-xl', a.value)}>
                    {value}
                    {unit && <span className="text-sm font-normal text-white/40 ml-1">{unit}</span>}
                </div>
                <div className="text-xs text-white/50 uppercase tracking-wider mt-1">{label}</div>
            </div>
        </div>
    );
}
