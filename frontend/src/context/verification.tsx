import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { VerifyResponse } from "../types/api";

interface CameraDevice {
  deviceId: string;
  label: string;
}

interface VerificationContextType {
  // Camera State
  videoRef: React.RefObject<HTMLVideoElement | null>;
  stream: MediaStream | null;
  isReady: boolean;
  error: string | null;
  capture: () => Promise<string>;
  captureSequence: (count: number, delayMs: number) => Promise<string[]>;

  // Camera Selection
  cameras: CameraDevice[];
  selectedCamera: string | null;
  switchCamera: (deviceId: string) => Promise<void>;
  refreshCameras: () => Promise<void>;

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
  
  // Camera Selection State
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  // Enumerate available cameras
  const refreshCameras = useCallback(async () => {
    try {
      // Need to request permission first to get device labels
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
      tempStream.getTracks().forEach(t => t.stop());
      
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices
        .filter(d => d.kind === "videoinput")
        .map((d, idx) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${idx + 1}`,
        }));
      setCameras(videoDevices);
      console.log("[CAMERA] Found cameras:", videoDevices);
    } catch (err) {
      console.error("[CAMERA] Failed to enumerate devices:", err);
    }
  }, []);

  // Start camera with specific deviceId
  const startCamera = useCallback(async (deviceId?: string) => {
    // Stop existing stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
    }
    
    const constraints: MediaStreamConstraints = {
      video: deviceId 
        ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
        : { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    };
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setError(null);
      setIsReady(true);
      
      // Update selected camera from actual stream
      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack) {
        const settings = videoTrack.getSettings();
        if (settings.deviceId) {
          setSelectedCamera(settings.deviceId);
        }
      }
      console.log("[CAMERA] Started camera:", deviceId || "default");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to access camera");
      setIsReady(false);
    }
  }, []);

  // Switch to a different camera
  const switchCamera = useCallback(async (deviceId: string) => {
    console.log("[CAMERA] Switching to:", deviceId);
    await startCamera(deviceId);
  }, [startCamera]);

  // Initial camera setup
  useEffect(() => {
    let mounted = true;
    
    const init = async () => {
      await refreshCameras();
      if (mounted) {
        await startCamera();
      }
    };
    
    init();

    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [refreshCameras, startCamera]);

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
      stream: streamRef.current,
      isReady,
      error,
      capture,
      captureSequence,
      cameras,
      selectedCamera,
      switchCamera,
      refreshCameras,
      lastCapture,
      lastResult,
      setLastCapture,
      setLastResult,
      history,
      addHistory,
    }),
    [isReady, error, capture, captureSequence, cameras, selectedCamera, switchCamera, refreshCameras, lastCapture, lastResult, history, addHistory]
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

