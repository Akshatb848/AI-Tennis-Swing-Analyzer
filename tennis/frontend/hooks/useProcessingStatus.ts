'use client';

import { useQuery } from '@tanstack/react-query';
import { recordingApi, RecordingStatus } from '@/lib/api';

export function useProcessingStatus(sessionId: string | null, enabled = true) {
    return useQuery<RecordingStatus>({
        queryKey: ['processing-status', sessionId],
        queryFn: async () => {
            if (!sessionId) throw new Error('No session id');
            const { data } = await recordingApi.status(sessionId);
            return data;
        },
        enabled: !!sessionId && enabled,
        refetchInterval: (query) => {
            const state = query.state.data?.state;
            // Stop polling once processing is complete or failed
            if (state === 'completed' || state === 'stopped' || state === 'failed') return false;
            return 2000; // 2 s
        },
        staleTime: 0,
    });
}
