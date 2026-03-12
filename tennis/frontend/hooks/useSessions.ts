'use client';

import { useQuery } from '@tanstack/react-query';
import { sessionsApi, Session } from '@/lib/api';

export interface SessionListResponse {
    sessions: Session[];
    total: number;
    page: number;
    page_size: number;
}

export function useSessions(page = 1, pageSize = 20) {
    return useQuery<SessionListResponse>({
        queryKey: ['sessions', page, pageSize],
        queryFn: async () => {
            const { data } = await sessionsApi.list(page, pageSize);
            return data as SessionListResponse;
        },
        staleTime: 30_000,
    });
}

export function useSession(sessionId: string | null) {
    return useQuery<Session>({
        queryKey: ['session', sessionId],
        queryFn: async () => {
            if (!sessionId) throw new Error('No session id');
            const { data } = await sessionsApi.get(sessionId);
            return data;
        },
        enabled: !!sessionId,
        staleTime: 10_000,
    });
}
