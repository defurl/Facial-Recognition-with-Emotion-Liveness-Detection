import { useCallback, useEffect, useRef, useState } from "react";
import { verifyFace } from "../services/apiClient";
import { useVerification } from "../context/verification";
import "../App.css";

interface VerificationPanelProps {
  isPaused?: boolean;
}

export function VerificationPanel({ isPaused = false }: VerificationPanelProps) {
  // Use shared Verification Context
  const { videoRef, isReady, error, captureSequence, lastResult, setLastCapture, setLastResult, addHistory } = useVerification();

  const [message, setMessage] = useState<string | null>(null);
  const [isMirrored, setIsMirrored] = useState(false);
  const [blinkChallenge, setBlinkChallenge] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastAttendanceRef = useRef<number>(0);

  // Rolling buffer: Keep last 20 frames for instant blink check
  const frameBufferRef = useRef<string[]>([]);
  const BUFFER_SIZE = 20;

  const handleVerify = useCallback(async () => {
    setMessage(null);
    try {
      // Capture single frame for fast verification
      const frames = await captureSequence(1, 0);
      const primaryFrame = frames[0];

      // Add to rolling buffer (FIFO)
      frameBufferRef.current.push(primaryFrame);
      if (frameBufferRef.current.length > BUFFER_SIZE) {
        frameBufferRef.current.shift();
      }

      // Check if we have enough frames for blink detection
      const hasEnoughFrames = frameBufferRef.current.length >= 10;

      // Use blink sequence if we have enough frames (allows instant blink check)
      // Always send mark_attendance=true - backend handles cooldown logic
      let payload = await verifyFace({
        image_b64: primaryFrame,
        blink_sequence: hasEnoughFrames ? frameBufferRef.current.slice(-15) : undefined,
        mark_attendance: true
      });

      // Show blink overlay if needed
      const needsBlink = payload.liveness === "Spoof" && (payload.blink?.blinks_needed ?? 0) > 0;
      setBlinkChallenge(needsBlink);

      setLastCapture(primaryFrame);
      setLastResult(payload);
      addHistory({
        timestamp: Date.now(),
        confidence: payload.confidence,
        distance: payload.distance,
        liveness: payload.liveness,
      });
    } catch (err) {
      // Suppress common transient errors to prevent flicker
      const errorMsg = err instanceof Error ? err.message : "Unable to verify";
      const suppressedErrors = [
        "No faces detected",
        "No employees registered",
        "Failed to fetch",
        "NetworkError",
        "Network request failed"
      ];
      const shouldSuppress = suppressedErrors.some(s => errorMsg.includes(s));

      // Clear detections on error so boxes don't persist when server is down or face is lost
      setLastResult(null);

      if (!shouldSuppress) {
        setMessage(errorMsg);
      }
    }
  }, [captureSequence, setLastCapture, setLastResult, addHistory]);

  const toggleMirror = () => setIsMirrored(prev => !prev);

  // Auto-verify loop (recursive with delay to prevent overlap)
  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let mounted = true;

    const loop = async () => {
      // Stop loop if paused or not ready
      if (!isReady || isPaused) return;

      try {
        await handleVerify();
      } catch (e) {
        console.error("Auto-verify error:", e);
      }

      if (mounted && !isPaused) {
        // Wait 100ms before next attempt (plus execution time of handleVerify)
        timeoutId = setTimeout(loop, 100);
      }
    };

    if (isReady && !isPaused) {
      loop();
    }

    return () => {
      mounted = false;
      clearTimeout(timeoutId);
    };
  }, [isReady, isPaused, handleVerify]);

  const drawDetections = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match resolution exactly
    const width = video.videoWidth;
    const height = video.videoHeight;
    // Prevent drawing if video not ready
    if (!width || !height) return;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.clearRect(0, 0, width, height);

    const detections = lastResult?.detections ?? [];

    // Draw detected faces
    detections.forEach((face) => {
      const { bbox, is_primary, identity, confidence, liveness } = face;

      const w = bbox.width * width;
      const h = bbox.height * height;
      const rawX = bbox.x * width;

      // Calculate X based on mirroring state
      // If Mirrored: width - rawX - w
      // If Normal: rawX
      const x = isMirrored ? (width - rawX - w) : rawX;
      const y = bbox.y * height;


      // Box styling: green if verified (identity present), orange if blink needed
      const isVerified = !!identity;

      const strokeColor = is_primary
        ? (isVerified ? "rgba(16, 185, 129, 0.9)" : "rgba(251, 146, 60, 0.9)")  // green : orange
        : "rgba(148, 163, 184, 0.5)";

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      if (is_primary) {
        // Label Background
        ctx.font = "600 14px 'Inter', sans-serif";

        // Simple display logic:
        // - If identity is present (verified/locked/already checked in) → show "✅ {name}"
        // - Otherwise → show "👁️ Blink to Verify"
        const idText = identity ? `✅ ${identity}` : "👁️ Blink to Verify";
        
        // const confText = identity && confidence ? `${Math.round(confidence)}%` : "";
        const livenessText = identity ? (liveness || "") : "";

        // // Blink Debug Info
        // const blink = lastResult?.blink;
        // const debugText = blink
        //   ? `EAR: ${blink.min_ear?.toFixed(2) ?? "--"} | Frames: ${blink.frames_processed} | Needed: ${blink.blinks_needed}`
        //   : "";

        // Identity Tag (Top)
        const idWidth = ctx.measureText(idText).width;
        ctx.fillStyle = strokeColor;
        ctx.fillRect(x, y - 28, idWidth + 20, 28);

        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "middle";
        ctx.fillText(idText, x + 10, y - 14);

        // Stats Tag (Bottom)
        const bottomY = y + h;
        const statsText = `${livenessText}`; //${confText}
        const statsWidth = ctx.measureText(statsText).width;

        ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
        ctx.fillRect(x, bottomY, statsWidth + 20, 28);

        ctx.fillStyle = isVerified ? "#86efac" : "#fca5a5";
        ctx.fillText(statsText, x + 10, bottomY + 14);

        // // Debug Info (Bottom + 30) - Always show if spoof or if debugText exists
        // if (debugText) {
        //   const debugWidth = ctx.measureText(debugText).width;
        //   ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        //   ctx.fillRect(x, bottomY + 30, debugWidth + 20, 24);
        //   ctx.fillStyle = "#cbd5e1";
        //   ctx.font = "500 12px 'Inter', monospace";
        //   ctx.fillText(debugText, x + 10, bottomY + 42);
        // }
      }
    });
  }, [lastResult, videoRef, isMirrored]);

  useEffect(() => {
    drawDetections();
  }, [drawDetections, isReady]);

  return (
    <article className="panel verification-panel">
      <div className="panel-header">
        <h3>Verification preview</h3>
        <button className="ghost-button" onClick={toggleMirror}>
          {isMirrored ? "Disable Mirror" : "Mirror Camera"}
        </button>
      </div>
      {error && <p className="muted">Camera error: {error}</p>}
      <div className="video-wrapper">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`preview-video ${isMirrored ? "mirrored" : ""}`}
        />
        <canvas ref={canvasRef} className="annotation-canvas" aria-hidden />
      </div>
      {message && <p className="muted error">{message}</p>}
    </article>
  );
}
