'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { useAuthStore } from '@/store/authStore';

const queryClientSingleton = new QueryClient({
    defaultOptions: {
        queries: {
            retry: 1,
            refetchOnWindowFocus: false,
        },
    },
});

export function Providers({ children }: { children: React.ReactNode }) {
    const queryClient = useRef(queryClientSingleton).current;
    const hydrate = useAuthStore((s) => s.hydrate);

    useEffect(() => {
        hydrate();
    }, [hydrate]);

    return (
        <QueryClientProvider client={queryClient}>
            {children}
        </QueryClientProvider>
    );
}
