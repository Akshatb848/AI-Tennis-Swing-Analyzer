'use client';

import { useState, useCallback } from 'react';
import { uploadVideoChunked, UploadChunkResponse } from '@/lib/api';

export type UploadState = 'idle' | 'uploading' | 'complete' | 'error';

export interface UseUploadReturn {
    state: UploadState;
    progress: number;
    result: UploadChunkResponse | null;
    error: string | null;
    upload: (file: File) => Promise<UploadChunkResponse | null>;
    reset: () => void;
}

export function useUpload(): UseUploadReturn {
    const [state, setState] = useState<UploadState>('idle');
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<UploadChunkResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const upload = useCallback(async (file: File): Promise<UploadChunkResponse | null> => {
        setState('uploading');
        setProgress(0);
        setError(null);
        setResult(null);

        try {
            const res = await uploadVideoChunked(file, setProgress);
            setResult(res);
            setState('complete');
            return res;
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : 'Upload failed';
            setError(msg);
            setState('error');
            return null;
        }
    }, []);

    const reset = useCallback(() => {
        setState('idle');
        setProgress(0);
        setResult(null);
        setError(null);
    }, []);

    return { state, progress, result, error, upload, reset };
}
