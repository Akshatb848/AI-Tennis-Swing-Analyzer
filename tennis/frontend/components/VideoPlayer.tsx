'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize, SkipBack, SkipForward } from 'lucide-react';
import clsx from 'clsx';

// ─── Overlay types ───────────────────────────────────────────────────────────

export interface CanvasDrawInstruction {
    type: 'coaching_marker' | 'arrow' | 'bounding_box' | 'speed_badge'
    | 'trajectory' | 'bounce' | 'linecall' | 'speed' | 'score';
    x?: number;     // 0–1
    y?: number;
    x1?: number; y1?: number;
    x2?: number; y2?: number;
    points?: { x: number; y: number }[];
    label?: string;
    color?: string;
}

export interface FrameOverlay {
    timestamp: number;       // seconds
    duration?: number;       // seconds to show (default 2.5)
    label: string;
    color: string;
    severity?: 'high' | 'medium' | 'low' | 'info';
    overlays: CanvasDrawInstruction[];
    coaching_text?: string;
    tip?: string;
    source?: string;
}

// Legacy simple overlay (kept for backward compat)
export interface Overlay {
    timestamp: number;
    duration?: number;
    label: string;
    type: 'bounce' | 'linecall' | 'speed' | 'score' | 'coaching_marker';
    x: number;
    y: number;
}

interface VideoPlayerProps {
    src: string;
    frameOverlays?: FrameOverlay[];
    overlays?: Overlay[];           // legacy
    onTimeUpdate?: (t: number) => void;
    onActiveOverlayChange?: (ov: FrameOverlay | null) => void;
    jumpTo?: number | null;
    className?: string;
}

function formatTime(s: number) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

function lerpAlpha(now: number, start: number, end: number): number {
    const fadeIn = 0.3;
    const fadeOut = 0.5;
    if (now < start + fadeIn) return (now - start) / fadeIn;
    if (now > end - fadeOut) return Math.max(0, (end - now) / fadeOut);
    return 1;
}

// ─── Canvas drawing helpers ───────────────────────────────────────────────────

function drawInstruction(
    ctx: CanvasRenderingContext2D,
    instr: CanvasDrawInstruction,
    w: number,
    h: number,
    alpha: number,
) {
    ctx.globalAlpha = alpha;
    const color = instr.color ?? '#f59e0b';

    switch (instr.type) {
        case 'coaching_marker': {
            const x = (instr.x ?? 0.5) * w;
            const y = (instr.y ?? 0.5) * h;
            // Pulsing circle
            ctx.beginPath();
            ctx.arc(x, y, 22, 0, Math.PI * 2);
            ctx.fillStyle = `${color}33`;
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.stroke();
            // Label pill
            if (instr.label) {
                ctx.font = 'bold 11px Inter, sans-serif';
                const tw = ctx.measureText(instr.label).width + 20;
                const pill_x = x - tw / 2;
                const pill_y = y + 26;
                ctx.fillStyle = '#0a0a0f';
                roundRect(ctx, pill_x, pill_y, tw, 22, 6);
                ctx.fill();
                ctx.fillStyle = color;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(instr.label, x, pill_y + 11);
            }
            break;
        }
        case 'speed_badge': {
            const x = (instr.x ?? 0.5) * w;
            const y = (instr.y ?? 0.5) * h;
            if (instr.label) {
                ctx.font = 'bold 12px Inter, sans-serif';
                const tw = ctx.measureText(instr.label).width + 16;
                ctx.fillStyle = '#059669cc';
                roundRect(ctx, x - tw / 2, y - 13, tw, 20, 10);
                ctx.fill();
                ctx.fillStyle = '#ffffff';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(instr.label, x, y - 3);
            }
            break;
        }
        case 'arrow': {
            const x1 = (instr.x1 ?? 0.4) * w;
            const y1 = (instr.y1 ?? 0.5) * h;
            const x2 = (instr.x2 ?? 0.6) * w;
            const y2 = (instr.y2 ?? 0.4) * h;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
            // Arrowhead
            const angle = Math.atan2(y2 - y1, x2 - x1);
            const size = 9;
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - size * Math.cos(angle - 0.4), y2 - size * Math.sin(angle - 0.4));
            ctx.lineTo(x2 - size * Math.cos(angle + 0.4), y2 - size * Math.sin(angle + 0.4));
            ctx.closePath();
            ctx.fillStyle = color;
            ctx.fill();
            break;
        }
        case 'bounding_box': {
            const x1 = (instr.x1 ?? 0.2) * w;
            const y1 = (instr.y1 ?? 0.2) * h;
            const x2 = (instr.x2 ?? 0.8) * w;
            const y2 = (instr.y2 ?? 0.8) * h;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 3]);
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);
            // Corner accents
            const cornerLen = 12;
            ctx.lineWidth = 3;
            [[x1, y1, 1, 1], [x2, y1, -1, 1], [x1, y2, 1, -1], [x2, y2, -1, -1]].forEach(([cx, cy, dx, dy]) => {
                ctx.beginPath();
                ctx.moveTo(cx + dx * cornerLen, cy as number);
                ctx.lineTo(cx, cy as number);
                ctx.lineTo(cx, (cy as number) + (dy as number) * cornerLen);
                ctx.strokeStyle = color;
                ctx.stroke();
            });
            break;
        }
        case 'trajectory': {
            if (instr.points && instr.points.length > 1) {
                ctx.beginPath();
                instr.points.forEach((p, i) => {
                    const px = p.x * w, py = p.y * h;
                    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                });
                ctx.strokeStyle = `${color}cc`;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            break;
        }
        default: {
            // Legacy simple circle for 'bounce' / 'speed' / etc.
            const x = (instr.x ?? 0.5) * w;
            const y = (instr.y ?? 0.5) * h;
            ctx.beginPath();
            ctx.arc(x, y, 14, 0, Math.PI * 2);
            ctx.fillStyle = `${color}33`;
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }
    ctx.globalAlpha = 1;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
}

// ─── Main VideoPlayer ─────────────────────────────────────────────────────────

export function VideoPlayer({
    src,
    frameOverlays = [],
    overlays = [],
    onTimeUpdate,
    onActiveOverlayChange,
    jumpTo,
    className,
}: VideoPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const animRef = useRef<number | null>(null);

    const [playing, setPlaying] = useState(false);
    const [muted, setMuted] = useState(false);
    const [progress, setProgress] = useState(0);
    const [duration, setDuration] = useState(0);
    const [currentTime, setCurrentTime] = useState(0);
    const [showControls, setShowControls] = useState(true);
    const controlsTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const lastActiveRef = useRef<string | null>(null);

    // Jump to timestamp
    useEffect(() => {
        if (jumpTo !== null && jumpTo !== undefined && videoRef.current) {
            videoRef.current.currentTime = jumpTo;
        }
    }, [jumpTo]);

    // Draw overlays on canvas (called every animation frame while playing)
    const drawFrame = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!canvas || !video) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.offsetWidth;
        canvas.height = video.offsetHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const now = video.currentTime;

        // Find active frame overlays
        let activeOverlay: FrameOverlay | null = null;
        for (const ov of frameOverlays) {
            const end = ov.timestamp + (ov.duration ?? 2.5);
            if (now >= ov.timestamp && now <= end) {
                const alpha = lerpAlpha(now, ov.timestamp, end);
                if (ov.overlays?.length) {
                    ov.overlays.forEach(instr => drawInstruction(ctx, instr, canvas.width, canvas.height, alpha));
                }
                activeOverlay = ov;
                break; // show one overlay at a time
            }
        }

        // Legacy simple overlays
        for (const ov of overlays) {
            const end = ov.timestamp + (ov.duration ?? 1.5);
            if (now < ov.timestamp || now > end) continue;
            const alpha = lerpAlpha(now, ov.timestamp, end);
            const colors: Record<string, string> = {
                bounce: '#22c55e', linecall: '#ef4444', speed: '#f59e0b',
                score: '#8b5cf6', coaching_marker: '#f59e0b',
            };
            drawInstruction(ctx, { type: ov.type, x: ov.x, y: ov.y, label: ov.label, color: colors[ov.type] ?? '#fff' },
                canvas.width, canvas.height, alpha);
        }

        // Notify parent about active overlay change
        const activeId = activeOverlay?.timestamp.toString() ?? null;
        if (activeId !== lastActiveRef.current) {
            lastActiveRef.current = activeId;
            onActiveOverlayChange?.(activeOverlay);
        }
    }, [frameOverlays, overlays, onActiveOverlayChange]);

    // Animation loop
    useEffect(() => {
        const loop = () => {
            drawFrame();
            animRef.current = requestAnimationFrame(loop);
        };
        animRef.current = requestAnimationFrame(loop);
        return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
    }, [drawFrame]);

    const handleTimeUpdate = useCallback(() => {
        const v = videoRef.current;
        if (!v) return;
        const t = v.currentTime;
        setCurrentTime(t);
        setProgress(v.duration ? t / v.duration : 0);
        onTimeUpdate?.(t);
    }, [onTimeUpdate]);

    const togglePlay = () => {
        const v = videoRef.current;
        if (!v) return;
        if (playing) { v.pause(); setPlaying(false); }
        else { v.play(); setPlaying(true); }
    };

    const toggleMute = () => {
        const v = videoRef.current;
        if (!v) return;
        v.muted = !v.muted;
        setMuted(!muted);
    };

    const seek = (e: React.MouseEvent<HTMLDivElement>) => {
        const v = videoRef.current;
        if (!v || !v.duration) return;
        const rect = e.currentTarget.getBoundingClientRect();
        v.currentTime = ((e.clientX - rect.left) / rect.width) * v.duration;
    };

    const skip = (delta: number) => {
        if (videoRef.current) videoRef.current.currentTime += delta;
    };

    const fullscreen = () => containerRef.current?.requestFullscreen?.();

    const handleMouseMove = () => {
        setShowControls(true);
        if (controlsTimer.current) clearTimeout(controlsTimer.current);
        controlsTimer.current = setTimeout(() => setShowControls(false), 2500);
    };

    // Severity color for scrubber markers
    const severityColors: Record<string, string> = {
        high: '#ef4444', medium: '#f59e0b', low: '#3b82f6', info: '#10b981',
    };

    return (
        <div
            ref={containerRef}
            className={clsx('relative bg-black rounded-2xl overflow-hidden group select-none', className)}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setShowControls(false)}
        >
            <video
                ref={videoRef}
                src={src}
                className="w-full h-full object-contain"
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={() => setDuration(videoRef.current?.duration ?? 0)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
                onClick={togglePlay}
                crossOrigin="anonymous"
            />

            {/* Canvas overlay — drawn on top of video */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 pointer-events-none"
                style={{ width: '100%', height: '100%' }}
            />

            {/* Coaching overlay instant indicator */}
            {frameOverlays.length > 0 && (
                <div className="absolute top-3 right-3 bg-black/50 rounded-lg px-2 py-1 flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                    <span className="text-xs text-white/60">{frameOverlays.length} coaching moments</span>
                </div>
            )}

            {/* Controls */}
            <div className={clsx(
                'absolute inset-0 flex flex-col justify-end transition-opacity duration-300',
                showControls || !playing ? 'opacity-100' : 'opacity-0',
            )}>
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent pointer-events-none" />

                <div className="relative px-4 pb-4 space-y-2.5">
                    {/* Scrubber with frame overlay markers */}
                    <div className="relative">
                        <div
                            className="w-full h-1.5 bg-white/20 rounded-full cursor-pointer hover:h-2 transition-all"
                            onClick={seek}
                        >
                            <div
                                className="h-full bg-violet-500 rounded-full relative"
                                style={{ width: `${progress * 100}%` }}
                            >
                                <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg" />
                            </div>
                        </div>
                        {/* Coaching moment dots on scrubber */}
                        {duration > 0 && frameOverlays.map((ov, i) => (
                            <div
                                key={i}
                                className="absolute top-0 w-1.5 h-1.5 rounded-full -translate-x-0.5 -translate-y-0"
                                style={{
                                    left: `${(ov.timestamp / duration) * 100}%`,
                                    background: severityColors[ov.severity ?? 'medium'],
                                }}
                                title={ov.label}
                            />
                        ))}
                    </div>

                    {/* Button row */}
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <button onClick={() => skip(-10)} className="text-white/70 hover:text-white transition-colors">
                                <SkipBack className="w-4 h-4" />
                            </button>
                            <button
                                onClick={togglePlay}
                                className="w-9 h-9 flex items-center justify-center bg-violet-600 hover:bg-violet-500 rounded-full transition-colors shadow-lg"
                            >
                                {playing
                                    ? <Pause className="w-4 h-4 text-white" fill="white" />
                                    : <Play className="w-4 h-4 text-white" fill="white" />}
                            </button>
                            <button onClick={() => skip(10)} className="text-white/70 hover:text-white transition-colors">
                                <SkipForward className="w-4 h-4" />
                            </button>
                            <span className="text-xs text-white/60 ml-2 font-mono">
                                {formatTime(currentTime)} / {formatTime(duration)}
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <button onClick={toggleMute} className="text-white/70 hover:text-white transition-colors">
                                {muted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                            </button>
                            <button onClick={fullscreen} className="text-white/70 hover:text-white transition-colors">
                                <Maximize className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
