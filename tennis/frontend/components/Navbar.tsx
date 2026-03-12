'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuthStore } from '@/store/authStore';
import { Activity, Upload, LayoutDashboard, User, LogOut, Menu, X } from 'lucide-react';
import { useState, useEffect } from 'react';
import clsx from 'clsx';

const NAV_LINKS = [
    { href: '/upload', label: 'Upload', icon: Upload },
    { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
];

export function Navbar() {
    const pathname = usePathname();
    const { user, isAuthenticated, logout } = useAuthStore();
    const [open, setOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => setScrolled(window.scrollY > 12);
        window.addEventListener('scroll', handleScroll, { passive: true });
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <header
            className={clsx(
                'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
                scrolled ? 'bg-surface-900/90 backdrop-blur-xl border-b border-white/[0.06] shadow-xl shadow-black/20' : 'bg-transparent',
            )}
        >
            <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
                {/* Logo */}
                <Link href="/" className="flex items-center gap-2.5 group">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-court-500 flex items-center justify-center shadow-lg shadow-brand-900/40 group-hover:shadow-brand-900/60 transition-all duration-300">
                        <Activity className="w-4 h-4 text-white" strokeWidth={2.5} />
                    </div>
                    <span className="font-bold text-white text-lg tracking-tight">TennisIQ</span>
                </Link>

                {/* Desktop links */}
                <div className="hidden md:flex items-center gap-1">
                    {NAV_LINKS.map(({ href, label, icon: Icon }) => (
                        <Link
                            key={href}
                            href={href}
                            className={clsx(
                                'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                                pathname === href
                                    ? 'bg-brand-600/20 text-brand-400'
                                    : 'text-white/60 hover:text-white hover:bg-white/5',
                            )}
                        >
                            <Icon className="w-4 h-4" />
                            {label}
                        </Link>
                    ))}
                </div>

                {/* Auth buttons */}
                <div className="hidden md:flex items-center gap-2">
                    {isAuthenticated && user ? (
                        <>
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-700 border border-white/10">
                                <User className="w-3.5 h-3.5 text-brand-400" />
                                <span className="text-sm text-white/80">{user.name}</span>
                                <span className="badge badge-brand capitalize">{user.subscriptionTier}</span>
                            </div>
                            <button onClick={logout} className="btn-ghost text-white/40 hover:text-red-400">
                                <LogOut className="w-4 h-4" />
                            </button>
                        </>
                    ) : (
                        <>
                            <Link href="/login" className="btn-ghost">Log in</Link>
                            <Link href="/register" className="btn-primary">Get started</Link>
                        </>
                    )}
                </div>

                {/* Mobile menu toggle */}
                <button
                    className="md:hidden btn-ghost"
                    onClick={() => setOpen((v) => !v)}
                    aria-label="Toggle menu"
                >
                    {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </button>
            </nav>

            {/* Mobile drawer */}
            {open && (
                <div className="md:hidden bg-surface-900/95 backdrop-blur-xl border-b border-white/[0.06] px-4 pb-4 animate-fade-in">
                    {NAV_LINKS.map(({ href, label, icon: Icon }) => (
                        <Link
                            key={href}
                            href={href}
                            onClick={() => setOpen(false)}
                            className={clsx(
                                'flex items-center gap-2 px-3 py-3 rounded-lg text-sm font-medium w-full transition-all',
                                pathname === href ? 'text-brand-400' : 'text-white/70',
                            )}
                        >
                            <Icon className="w-4 h-4" />
                            {label}
                        </Link>
                    ))}
                    <div className="mt-3 pt-3 border-t border-white/[0.06] flex flex-col gap-2">
                        {isAuthenticated ? (
                            <button onClick={() => { logout(); setOpen(false); }} className="btn-secondary w-full">
                                Log out
                            </button>
                        ) : (
                            <>
                                <Link href="/login" onClick={() => setOpen(false)} className="btn-secondary w-full text-center">Log in</Link>
                                <Link href="/register" onClick={() => setOpen(false)} className="btn-primary w-full text-center">Get started</Link>
                            </>
                        )}
                    </div>
                </div>
            )}
        </header>
    );
}
