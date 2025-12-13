import { useCallback, useEffect, useRef, useState } from "react";

const DEFAULT_CONSTRAINTS: MediaStreamConstraints = {
  video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
  audio: false,
};

export function useCamera(constraints: MediaStreamConstraints = DEFAULT_CONSTRAINTS) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setIsReady(false);
  }, []);

  const start = useCallback(() => {
    if (streamRef.current) {
      return;
    }
    navigator.mediaDevices
      .getUserMedia(constraints)
      .then((stream) => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setError(null);
        setIsReady(true);
      })
      .catch((err) => {
        setError(err.message ?? "Unable to access camera");
      });
  }, [constraints]);

  useEffect(() => {
    start();
    return stop;
  }, [start, stop]);

  const capture = useCallback(async () => {
    if (!videoRef.current) {
      throw new Error("Camera is not initialized");
    }
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Unable to create capture context");
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.8);
  }, []);

  return {
    videoRef,
    isReady,
    error,
    start,
    stop,
    capture,
  };
}
