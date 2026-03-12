'use client';

import { useQuery } from '@tanstack/react-query';
import axios, { AxiosError } from 'axios';

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export interface AnalysisStatus {
    job_id: string;
    session_id: string;
    state: 'queued' | 'running' | 'complete' | 'failed';
    stage_index: number;
    stage_name: string;
    stages: { id: string; label: string }[];
    progress: number;
    error: string | null;
    duration_seconds: number;
    fps: number;
    frames_total: number;
    frames_processed: number;
    players_detected: number;
    ball_detections: number;
    points_detected: number;
    line_calls: number;
}

const TERMINAL = new Set(['complete', 'failed']);

async function fetchStatus(jobId: string): Promise<AnalysisStatus> {
    const { data } = await axios.get<AnalysisStatus>(
        `${BACKEND}/api/v1/analyze/${jobId}`,
    );
    return data;
}

/**
 * Polls GET /api/v1/analyze/{jobId} every 1.5s.
 * Stops polling when:
 *   - job state is 'complete' or 'failed'
 *   - backend returns 404 (job doesn't exist — e.g. server restarted)
 *   - 3 consecutive network errors
 */
export function useAnalysisStatus(jobId: string | null) {
    const query = useQuery<AnalysisStatus>({
        queryKey: ['analysis-status', jobId],
        queryFn: () => fetchStatus(jobId!),
        enabled: !!jobId,

        // Stop polling on terminal state
        refetchInterval: (query) => {
            const data = query.state.data;
            const err = query.state.error as AxiosError | null;

            // Stop polling if we got data and it's terminal
            if (data && TERMINAL.has(data.state)) return false;

            // Stop polling on 404 — job doesn't exist (server restarted, stale URL)
            if (err && (err as AxiosError)?.response?.status === 404) return false;

            return 1500;
        },

        // Don't retry 404s — the job simply doesn't exist
        retry: (failureCount, error) => {
            const status = (error as AxiosError)?.response?.status;
            if (status === 404) return false;          // never retry 404
            return failureCount < 2;                   // retry network errors twice
        },

        // Keep stale data visible while refetching
        staleTime: 1000,
    });

    // Derived: is this a "job not found" error?
    const isNotFound =
        query.isError &&
        (query.error as AxiosError)?.response?.status === 404;

    return { ...query, isNotFound };
}
