'use client';

import { useCallback, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useUpload } from '@/hooks/useUpload';
import { Upload, FileVideo, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';
import clsx from 'clsx';

const ACCEPTED = ['.mp4', '.mov', '.mkv', '.m4v'];

function fileSizeMB(bytes: number) {
    return (bytes / 1024 / 1024).toFixed(1);
}

export default function UploadPage() {
    const router = useRouter();
    const { state, progress, result, error, upload } = useUpload();
    const [dragging, setDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleFile = useCallback(async (file: File) => {
        setSelectedFile(file);
        const res = await upload(file);
        if (res?.path) {
            // Encode path into query param for configure page
            router.push(`/configure?upload_id=${res.upload_id}&path=${encodeURIComponent(res.path!)}&filename=${encodeURIComponent(file.name)}`);
        }
    }, [upload, router]);

    const onDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleFile(file);
    }, [handleFile]);

    const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleFile(file);
    };

    return (
        <div className="max-w-2xl mx-auto px-4 py-16">
            {/* Stepper */}
            <div className="flex items-center gap-3 mb-10">
                {['Upload', 'Configure', 'Processing', 'Results'].map((step, i) => (
                    <div key={step} className="flex items-center gap-2">
                        {i > 0 && <div className="w-8 h-px bg-white/10" />}
                        <div className={clsx('step-dot', i === 0 ? 'step-dot-active' : 'step-dot-pending')}>
                            {i + 1}
                        </div>
                        <span className={clsx('text-xs font-medium', i === 0 ? 'text-white' : 'text-white/30')}>{step}</span>
                    </div>
                ))}
            </div>

            <h1 className="text-3xl font-bold text-white mb-2">Upload match video</h1>
            <p className="text-white/50 mb-10">
                Record from behind the baseline for best results. Supports MP4, MOV, MKV.
            </p>

            {/* Drop zone */}
            {state === 'idle' && (
                <label
                    htmlFor="video-upload"
                    className={clsx(
                        'group relative flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-16 cursor-pointer transition-all duration-300',
                        dragging
                            ? 'border-brand-400 bg-brand-600/10'
                            : 'border-white/10 bg-surface-800 hover:border-brand-500/50 hover:bg-brand-600/5',
                    )}
                    onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={onDrop}
                >
                    <div className="w-16 h-16 rounded-2xl bg-brand-600/15 flex items-center justify-center group-hover:scale-105 transition-transform duration-300">
                        <Upload className="w-8 h-8 text-brand-400" />
                    </div>
                    <div className="text-center">
                        <p className="text-white font-medium">Drag & drop your video here</p>
                        <p className="text-white/40 text-sm mt-1">or <span className="text-brand-400 underline underline-offset-2">browse files</span></p>
                    </div>
                    <p className="text-xs text-white/30">MP4 · MOV · MKV · up to 2 GB</p>
                    <input
                        id="video-upload"
                        type="file"
                        className="sr-only"
                        accept={ACCEPTED.join(',')}
                        onChange={onFileChange}
                    />
                </label>
            )}

            {/* Uploading state */}
            {state === 'uploading' && selectedFile && (
                <div className="card space-y-6">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-xl bg-surface-700 flex items-center justify-center shrink-0">
                            <FileVideo className="w-6 h-6 text-brand-400" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-white truncate">{selectedFile.name}</p>
                            <p className="text-xs text-white/40">{fileSizeMB(selectedFile.size)} MB</p>
                        </div>
                        <span className="badge badge-brand">{progress}%</span>
                    </div>
                    <div>
                        <div className="flex justify-between text-xs text-white/40 mb-2">
                            <span>Uploading…</span>
                            <span>{progress}%</span>
                        </div>
                        <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${progress}%` }} />
                        </div>
                    </div>
                    <p className="text-xs text-white/30 text-center">
                        Large files are uploaded in chunks. Do not close this window.
                    </p>
                </div>
            )}

            {/* Complete state */}
            {state === 'complete' && (
                <div className="card flex items-center gap-4">
                    <CheckCircle2 className="w-8 h-8 text-court-400 shrink-0" />
                    <div>
                        <p className="font-semibold text-white">Upload complete</p>
                        <p className="text-sm text-white/40">Redirecting to configuration…</p>
                    </div>
                </div>
            )}

            {/* Error state */}
            {state === 'error' && (
                <div className="card border-red-500/20 bg-red-500/5 flex items-start gap-4">
                    <AlertCircle className="w-6 h-6 text-red-400 shrink-0 mt-0.5" />
                    <div>
                        <p className="font-semibold text-red-400">Upload failed</p>
                        <p className="text-sm text-white/50 mt-1">{error}</p>
                        <button
                            onClick={() => { setSelectedFile(null); }}
                            className="btn-secondary mt-4"
                        >
                            Try again
                        </button>
                    </div>
                </div>
            )}

            {/* Tips */}
            <div className="mt-8 space-y-2">
                {TIPS.map((t) => (
                    <div key={t} className="flex items-start gap-2 text-xs text-white/30">
                        <ArrowRight className="w-3 h-3 mt-0.5 shrink-0 text-brand-500/50" />
                        {t}
                    </div>
                ))}
            </div>
        </div>
    );
}

const TIPS = [
    'Mount your device behind the baseline at shoulder height for best angle.',
    'Ensure the full court is visible in the frame as much as possible.',
    'Recording from a fixed position gives more reliable ball tracking.',
    'Longer matches may take a few minutes to process.',
];
