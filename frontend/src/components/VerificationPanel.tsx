import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { verifyFace } from "../services/apiClient";
import { useCamera } from "../hooks/useCamera";
import { useVerification } from "../context/verification";
import "../App.css";

export function VerificationPanel() {
  const { videoRef, isReady, error, capture } = useCamera();
  const { lastResult, setLastCapture, setLastResult } = useVerification();
  const [status, setStatus] = useState<"idle" | "capturing" | "error" | "success">("idle");
  const [message, setMessage] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const handleVerify = async () => {
    setStatus("capturing");
    setMessage(null);
    try {
      const image = await capture();
      const payload = await verifyFace({ image });
      setLastCapture(image);
      setLastResult(payload);
      setStatus("success");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to verify");
      setStatus("error");
    }
  };

  const livenessLabel = useMemo(() => {
    if (!lastResult) return "waiting";
    const normalized = lastResult.liveness.toLowerCase();
    if (normalized.includes("real")) return "live";
    if (normalized.includes("spoof")) return "spoof?";
    return normalized;
  }, [lastResult]);

  const drawDetections = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    canvas.width = width;
    canvas.height = height;
    ctx.clearRect(0, 0, width, height);

    const detections = lastResult?.detections ?? [];
    if (!detections.length) {
      return;
    }

    ctx.font = "500 12px 'Inter', system-ui, sans-serif";
    ctx.textBaseline = "top";

    detections.forEach((face) => {
      const { bbox, is_primary, identity } = face;
      const x = bbox.x * width;
      const y = bbox.y * height;
      const w = bbox.width * width;
      const h = bbox.height * height;
      const strokeColor = is_primary ? "rgba(16, 185, 129, 0.9)" : "rgba(248, 250, 252, 0.6)";
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = is_primary ? 3 : 2;
      ctx.strokeRect(x, y, w, h);

      const label = identity || (is_primary ? "Primary" : "Candidate");
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(2, 6, 23, 0.85)";
      ctx.fillRect(x, y - 24, textWidth + 16, 20);
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(label, x + 8, y - 22);
    });
  }, [lastResult, videoRef]);

  useEffect(() => {
    drawDetections();
  }, [drawDetections, isReady]);

  return (
    <article className="panel verification-panel">
      <div className="panel-header">
        <h3>Verification preview</h3>
        <button className="ghost-button" onClick={handleVerify} disabled={!isReady || status === "capturing"}>
          {status === "capturing" ? "Capturing…" : "Verify snapshot"}
        </button>
      </div>
      {error && <p className="muted">Camera error: {error}</p>}
      <div className="video-wrapper">
        <video ref={videoRef} autoPlay playsInline muted className="preview-video" />
        <div className={`status-pill ${lastResult ? (livenessLabel === "live" ? "ok" : "unavailable") : "waiting"}`}>
          {lastResult ? `${lastResult.identity ?? "Unknown"} · ${livenessLabel}` : "No verification yet"}
        </div>
        <canvas ref={canvasRef} className="annotation-canvas" aria-hidden />
      </div>
      {message && <p className="muted error">{message}</p>}
      {lastResult && (
        <dl className="result-grid">
          <div>
            <dt>Identity</dt>
            <dd>{lastResult.identity ?? "Unknown"}</dd>
          </div>
          <div>
            <dt>Distance</dt>
            <dd>{lastResult.distance.toFixed(3)}</dd>
          </div>
          <div>
            <dt>Confidence</dt>
            <dd>{(lastResult.confidence * 100).toFixed(1)}%</dd>
          </div>
          <div>
            <dt>Liveness</dt>
            <dd>{lastResult.liveness}</dd>
          </div>
        </dl>
      )}
    </article>
  );
}
