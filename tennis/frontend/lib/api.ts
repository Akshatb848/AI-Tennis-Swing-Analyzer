import axios from 'axios';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

const api = axios.create({
    baseURL: `${BASE_URL}/api/v1`,
    timeout: 60_000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Attach auth token from localStorage
api.interceptors.request.use((config) => {
    if (typeof window !== 'undefined') {
        const token = localStorage.getItem('tiq_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
    }
    return config;
});

// ── Auth ─────────────────────────────────────────────────────────────────────

export interface RegisterPayload {
    email: string;
    password: string;
    name: string;
}

export interface LoginPayload {
    email: string;
    password: string;
}

export interface AuthResponse {
    access_token: string;
    user_id: string;
    email: string;
    name: string;
    subscription_tier: string;
}

export const authApi = {
    register: (p: RegisterPayload) => api.post<AuthResponse>('/auth/register', p),
    login: (p: LoginPayload) => api.post<AuthResponse>('/auth/login', p),
    google: (idToken: string) => api.post<AuthResponse>('/auth/google', { id_token: idToken }),
    me: (userId: string) => api.get(`/auth/me?user_id=${userId}`),
};

// ── Upload ────────────────────────────────────────────────────────────────────

export interface UploadChunkResponse {
    upload_id: string;
    status: 'uploading' | 'complete';
    received?: number;
    total?: number;
    path?: string;
    size_bytes?: number;
    filename?: string;
}

const CHUNK_SIZE = 4 * 1024 * 1024; // 4 MB

export async function uploadVideoChunked(
    file: File,
    onProgress: (pct: number) => void,
): Promise<UploadChunkResponse> {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    let uploadId: string | null = null;

    for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const chunk = file.slice(start, start + CHUNK_SIZE);
        const form = new FormData();
        form.append('file', new File([chunk], file.name));
        form.append('chunk_index', String(i));
        form.append('total_chunks', String(totalChunks));
        if (uploadId) form.append('upload_id', uploadId);

        const { data } = await api.post<UploadChunkResponse>('/upload', form, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        uploadId = data.upload_id;
        onProgress(Math.round(((i + 1) / totalChunks) * 100));

        if (data.status === 'complete') return data;
    }

    // Should not reach here
    throw new Error('Upload did not complete');
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export interface Session {
    id: string;
    mode: string;
    status: string;
    match_type: string;
    environment: string;
    player_names: string[];
    court_surface: string;
    processing_progress: number;
    created_at: string;
    updated_at: string;
}

export const sessionsApi = {
    list: (page = 1, pageSize = 20) =>
        api.get('/sessions', { params: { page, page_size: pageSize } }),
    get: (id: string) => api.get<Session>(`/sessions/${id}`),
    create: (payload: Partial<Session>) => api.post<Session>('/sessions', null, { params: payload }),
    updateStatus: (id: string, status: string) =>
        api.patch(`/sessions/${id}/status`, null, { params: { status } }),
    delete: (id: string) => api.delete(`/sessions/${id}`),
};

// ── Recording ─────────────────────────────────────────────────────────────────

export interface RecordingSetupPayload {
    match_type?: string;
    environment?: string;
    player_names?: string[];
    court_surface?: string;
}

export interface RecordingStatus {
    session_id: string;
    state: string;
    frames_processed: number;
    points_detected: number;
    line_calls: number;
    duration_seconds: number;
    pipeline?: {
        ball_detections: number;
        player_detections: number;
        avg_latency_ms: number;
        is_running: boolean;
    };
}

export const recordingApi = {
    setup: (p: RecordingSetupPayload) =>
        api.post('/recording/setup', null, { params: p }),
    start: (sessionId: string) =>
        api.post(`/recording/${sessionId}/start`),
    stop: (sessionId: string) =>
        api.post(`/recording/${sessionId}/stop`),
    status: (sessionId: string) =>
        api.get<RecordingStatus>(`/recording/${sessionId}/status`),
    summary: (sessionId: string) =>
        api.get(`/recording/${sessionId}/summary`),
    ingestVideo: (sessionId: string, filePath: string) => {
        // Trigger ingestion of already-uploaded file
        return api.post(`/recording/${sessionId}/ingest-video`, null, {
            params: { file_path: filePath },
        });
    },
    liveState: (sessionId: string) =>
        api.get(`/recording/${sessionId}/live`),
};

// ── Matches ───────────────────────────────────────────────────────────────────

export const matchesApi = {
    create: (p: {
        player1_name?: string;
        player2_name?: string;
        match_format?: string;
        surface?: string;
        no_ad?: boolean;
    }) => api.post('/matches', null, { params: p }),
    get: (matchId: string) => api.get(`/matches/${matchId}`),
    getScore: (matchId: string) => api.get(`/matches/${matchId}/score`),
    getTimeline: (matchId: string) => api.get(`/matches/${matchId}/timeline`),
};

// ── Stats ─────────────────────────────────────────────────────────────────────

export const statsApi = {
    getSessionStats: (sessionId: string) =>
        api.get(`/stats/session/${sessionId}`),
};

// ── Subscriptions ─────────────────────────────────────────────────────────────

export const subscriptionApi = {
    getCurrent: () => api.get('/subscriptions/me'),
    getPlans: () => api.get('/subscriptions/plans'),
};

export default api;
