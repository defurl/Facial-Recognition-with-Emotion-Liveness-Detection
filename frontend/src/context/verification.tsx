import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { VerifyResponse } from "../types/api";

const DEFAULT_CONSTRAINTS: MediaStreamConstraints = {
  video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
  audio: false,
};

interface VerificationContextType {
  // Camera State
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isReady: boolean;
  error: string | null;
  capture: () => Promise<string>;
  captureSequence: (count: number, delayMs: number) => Promise<string[]>;

  // Verification State
  lastCapture: string | null;
  lastResult: VerifyResponse | null;
  setLastCapture: (value: string | null) => void;
  setLastResult: (value: VerifyResponse | null) => void;
  history: VerifyHistoryEntry[];
  addHistory: (entry: VerifyHistoryEntry) => void;
}

const VerificationContext = createContext<VerificationContextType | undefined>(undefined);

export interface VerifyHistoryEntry {
  timestamp: number;
  distance: number;
  confidence: number;
  liveness: string;
}

const HISTORY_LIMIT = 12;

export function VerificationProvider({ children }: { children: ReactNode }) {
  // Camera Logic
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    navigator.mediaDevices
      .getUserMedia(DEFAULT_CONSTRAINTS)
      .then((stream) => {
        if (!mounted) {
          stream.getTracks().forEach(t => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setError(null);
        setIsReady(true);
      })
      .catch((err) => {
        if (mounted) setError(err.message ?? "Unable to access camera");
      });

    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  const capture = useCallback(async () => {
    if (!videoRef.current) throw new Error("Camera not initialized");
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Context creation failed");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.8);
  }, []);

  const captureSequence = useCallback(async (count: number, delayMs: number) => {
    const frames: string[] = [];
    for (let i = 0; i < count; i++) {
      frames.push(await capture());
      if (i < count - 1) {
        await new Promise(r => setTimeout(r, delayMs));
      }
    }
    return frames;
  }, [capture]);

  // Verification Logic
  const [lastCapture, setLastCapture] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<VerifyResponse | null>(null);
  const [history, setHistory] = useState<VerifyHistoryEntry[]>([]);

  const addHistory = useCallback((entry: VerifyHistoryEntry) => {
    setHistory((prev) => {
      const next = [...prev, entry];
      return next.length > HISTORY_LIMIT ? next.slice(next.length - HISTORY_LIMIT) : next;
    });
  }, []);

  const value = useMemo(
    () => ({
      videoRef,
      isReady,
      error,
      capture,
      captureSequence,
      lastCapture,
      lastResult,
      setLastCapture,
      setLastResult,
      history,
      addHistory,
    }),
    [isReady, error, capture, captureSequence, lastCapture, lastResult, history, addHistory]
  );

  return <VerificationContext.Provider value={value}>{children}</VerificationContext.Provider>;
}

export function useVerification() {
  const context = useContext(VerificationContext);
  if (!context) {
    throw new Error("useVerification must be used within VerificationProvider");
  }
  return context;
}
