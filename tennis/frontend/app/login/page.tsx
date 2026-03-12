'use client';

import { useState, FormEvent, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/store/authStore';
import { Activity, Loader2 } from 'lucide-react';

declare global {
    interface Window {
        google?: {
            accounts: {
                id: {
                    initialize: (config: Record<string, unknown>) => void;
                    renderButton: (el: HTMLElement, config: Record<string, unknown>) => void;
                };
            };
        };
    }
}

const GOOGLE_CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? '';

export default function LoginPage() {
    const router = useRouter();
    const login = useAuthStore((s) => s.login);

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [googleLoading, setGoogleLoading] = useState(false);

    const handleLogin = (data: { access_token: string; user_id: string; email: string; name: string; subscription_tier: string }) => {
        login(data.access_token, {
            id: data.user_id,
            email: data.email,
            name: data.name,
            subscriptionTier: data.subscription_tier as 'free' | 'pro' | 'elite',
        });
        router.push('/dashboard');
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setSubmitting(true);
        setError(null);
        try {
            const { data } = await authApi.login({ email, password });
            handleLogin(data);
        } catch (err: unknown) {
            const axiosErr = err as { response?: { data?: { detail?: string } } };
            setError(axiosErr.response?.data?.detail ?? 'Invalid email or password.');
            setSubmitting(false);
        }
    };

    const handleGoogleCallback = async (response: { credential: string }) => {
        setGoogleLoading(true);
        setError(null);
        try {
            const { data } = await authApi.google(response.credential);
            handleLogin(data);
        } catch (err: unknown) {
            const axiosErr = err as { response?: { data?: { detail?: string } } };
            setError(axiosErr.response?.data?.detail ?? 'Google sign-in failed.');
            setGoogleLoading(false);
        }
    };

    // Load Google Identity Services
    useEffect(() => {
        if (!GOOGLE_CLIENT_ID) return;

        const script = document.createElement('script');
        script.src = 'https://accounts.google.com/gsi/client';
        script.async = true;
        script.defer = true;
        script.onload = () => {
            window.google?.accounts.id.initialize({
                client_id: GOOGLE_CLIENT_ID,
                callback: handleGoogleCallback,
            });
            const btnContainer = document.getElementById('google-signin-btn');
            if (btnContainer) {
                window.google?.accounts.id.renderButton(btnContainer, {
                    type: 'standard',
                    theme: 'filled_black',
                    size: 'large',
                    width: '100%',
                    text: 'signin_with',
                    shape: 'pill',
                });
            }
        };
        document.head.appendChild(script);
        return () => { script.remove(); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center px-4">
            <div className="w-full max-w-sm">
                <div className="flex items-center justify-center gap-2 mb-8">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-500 to-court-500 flex items-center justify-center">
                        <Activity className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-bold text-white text-xl">TennisIQ</span>
                </div>

                <div className="card">
                    <h1 className="text-xl font-bold text-white mb-1">Welcome back</h1>
                    <p className="text-white/40 text-sm mb-6">Sign in to your account</p>

                    {/* Google Sign-In */}
                    {GOOGLE_CLIENT_ID && (
                        <>
                            <div id="google-signin-btn" className="w-full mb-4 flex justify-center" />
                            {googleLoading && (
                                <div className="flex items-center justify-center gap-2 text-white/40 text-sm mb-4">
                                    <Loader2 className="w-4 h-4 animate-spin" /> Signing in with Google…
                                </div>
                            )}
                            <div className="flex items-center gap-3 mb-4">
                                <div className="flex-1 h-px bg-white/10" />
                                <span className="text-xs text-white/30 uppercase">or</span>
                                <div className="flex-1 h-px bg-white/10" />
                            </div>
                        </>
                    )}

                    {/* Email + Password */}
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="label">Email</label>
                            <input
                                type="email"
                                className="input"
                                placeholder="you@example.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>
                        <div>
                            <label className="label">Password</label>
                            <input
                                type="password"
                                className="input"
                                placeholder="••••••••"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>

                        {error && (
                            <p className="text-red-400 text-sm">{error}</p>
                        )}

                        <button type="submit" disabled={submitting} className="btn-primary w-full py-3">
                            {submitting ? <><Loader2 className="w-4 h-4 animate-spin" /> Signing in…</> : 'Sign in'}
                        </button>
                    </form>

                    <p className="text-center text-sm text-white/40 mt-5">
                        No account?{' '}
                        <Link href="/register" className="text-brand-400 hover:text-brand-300 underline underline-offset-2">
                            Create one
                        </Link>
                    </p>
                </div>
            </div>
        </div>
    );
}
