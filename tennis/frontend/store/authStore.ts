import { create } from 'zustand';

interface User {
    id: string;
    email: string;
    name: string;
    subscriptionTier: 'free' | 'pro' | 'elite';
}

interface AuthState {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
    login: (token: string, user: User) => void;
    logout: () => void;
    hydrate: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
    user: null,
    token: null,
    isAuthenticated: false,

    login: (token, user) => {
        if (typeof window !== 'undefined') {
            localStorage.setItem('tiq_token', token);
            localStorage.setItem('tiq_user', JSON.stringify(user));
        }
        set({ token, user, isAuthenticated: true });
    },

    logout: () => {
        if (typeof window !== 'undefined') {
            localStorage.removeItem('tiq_token');
            localStorage.removeItem('tiq_user');
        }
        set({ token: null, user: null, isAuthenticated: false });
    },

    hydrate: () => {
        if (typeof window === 'undefined') return;
        const token = localStorage.getItem('tiq_token');
        const raw = localStorage.getItem('tiq_user');
        if (token && raw) {
            try {
                const user = JSON.parse(raw) as User;
                set({ token, user, isAuthenticated: true });
            } catch {
                localStorage.removeItem('tiq_token');
                localStorage.removeItem('tiq_user');
            }
        }
    },
}));
