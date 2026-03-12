'use client';

import { useEffect, useRef, useCallback, useState } from 'react';

interface VoiceCue {
    timestamp: number;
    text: string;
    event_type: string;
}

interface UseVoiceCoachOptions {
    script: VoiceCue[];
    currentTime: number;
    enabled: boolean;
    rate?: number;   // speech rate 0.5-2.0 (default 0.9)
    pitch?: number;  // 0-2 (default 1.0)
}

interface UseVoiceCoachReturn {
    speak: (text: string) => void;
    stop: () => void;
    isSpeaking: boolean;
    isSupported: boolean;
}

const LOOKAHEAD = 0.3; // seconds before timestamp to trigger cue

export function useVoiceCoach({
    script,
    currentTime,
    enabled,
    rate = 0.9,
    pitch = 1.0,
}: UseVoiceCoachOptions): UseVoiceCoachReturn {
    const [isSpeaking, setIsSpeaking] = useState(false);
    const firedRef = useRef<Set<number>>(new Set());
    const synthRef = useRef<SpeechSynthesis | null>(null);
    const isSupported = typeof window !== 'undefined' && 'speechSynthesis' in window;

    useEffect(() => {
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
            synthRef.current = window.speechSynthesis;
        }
    }, []);

    const speak = useCallback((text: string) => {
        if (!synthRef.current || !text) return;
        synthRef.current.cancel();
        const utt = new SpeechSynthesisUtterance(text);
        utt.rate = rate;
        utt.pitch = pitch;
        utt.volume = 1;
        // Prefer a natural-sounding voice
        const voices = synthRef.current.getVoices();
        const preferred = voices.find(v =>
            v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha'))
        ) ?? voices.find(v => v.lang.startsWith('en'));
        if (preferred) utt.voice = preferred;
        utt.onstart = () => setIsSpeaking(true);
        utt.onend = () => setIsSpeaking(false);
        utt.onerror = () => setIsSpeaking(false);
        synthRef.current.speak(utt);
    }, [rate, pitch]);

    const stop = useCallback(() => {
        synthRef.current?.cancel();
        setIsSpeaking(false);
    }, []);

    // Auto-trigger cues from script based on currentTime
    useEffect(() => {
        if (!enabled || !script.length || !isSupported) return;

        for (const cue of script) {
            const fireAt = cue.timestamp - LOOKAHEAD;
            if (
                currentTime >= fireAt &&
                currentTime < cue.timestamp + 1.0 && // don't re-fire late
                !firedRef.current.has(cue.timestamp)
            ) {
                firedRef.current.add(cue.timestamp);
                speak(cue.text);
                break; // one cue at a time
            }
        }
    }, [currentTime, script, enabled, speak, isSupported]);

    // Reset fired cues when script changes (new video)
    useEffect(() => {
        firedRef.current.clear();
    }, [script]);

    return { speak, stop, isSpeaking, isSupported };
}
