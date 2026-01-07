import { useEffect, useRef, useState, useCallback } from "react";
import { useVerification } from "../context/verification";
import { resetLivenessCache } from "../services/apiClient";
import "../App.css";

interface EARDebugPanelProps {
  className?: string;
}

/**
 * EAR Debug Panel - Shows real-time Eye Aspect Ratio visualization
 * Similar to the Python desktop app's BLINK DETECTION DEBUG panel
 */
export function EARDebugPanel({ className = "" }: EARDebugPanelProps) {
  const { lastResult } = useVerification();
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const earHistoryRef = useRef<number[]>([]);
  const [resetStatus, setResetStatus] = useState<string | null>(null);

  const HISTORY_SIZE = 50; // Number of EAR values to display

  // Extract blink data
  const blink = lastResult?.blink;
  const currentEar = blink?.current_ear;
  const minEar = blink?.min_ear;
  const maxEar = blink?.max_ear;
  const framesProcessed = blink?.frames_processed ?? 0;
  const blinksNeeded = blink?.blinks_needed ?? 1;
  const hasBlinked = blink?.has_blinked ?? false;
  // Use dynamic threshold from backend, default to 0.35
  const threshold = blink?.threshold ?? 0.35;

  // Update EAR history when new result arrives
  useEffect(() => {
    if (currentEar != null) {
      earHistoryRef.current.push(currentEar);
      if (earHistoryRef.current.length > HISTORY_SIZE) {
        earHistoryRef.current.shift();
      }
    }
  }, [currentEar]);

  // Draw the EAR visualization
  const drawEARGraph = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const earHistory = earHistoryRef.current;

    // Clear canvas
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw threshold line
    const thresholdY = height - (threshold / 0.8) * height;
    ctx.strokeStyle = "#d29922";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(0, thresholdY);
    ctx.lineTo(width, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label for threshold
    ctx.fillStyle = "#d29922";
    ctx.font = "10px Inter, sans-serif";
    ctx.fillText(`Threshold: ${threshold}`, 5, thresholdY - 5);

    // Draw EAR values
    if (earHistory.length > 1) {
      ctx.strokeStyle = "#3fb950";
      ctx.lineWidth = 2;
      ctx.beginPath();

      const step = width / (HISTORY_SIZE - 1);
      for (let i = 0; i < earHistory.length; i++) {
        const x = i * step;
        const y = height - (earHistory[i] / 0.8) * height;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Draw current EAR point
      const lastEar = earHistory[earHistory.length - 1];
      const lastX = (earHistory.length - 1) * step;
      const lastY = height - (lastEar / 0.8) * height;

      ctx.fillStyle = lastEar < threshold ? "#f85149" : "#3fb950";
      ctx.beginPath();
      ctx.arc(lastX, lastY, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [threshold]);

  // Redraw on each frame
  useEffect(() => {
    drawEARGraph();
  }, [lastResult, drawEARGraph]);

  // Handle reset button
  const handleReset = async () => {
    try {
      setResetStatus("Resetting...");
      const result = await resetLivenessCache();
      setResetStatus(result.message);
      setTimeout(() => setResetStatus(null), 3000);
    } catch (error) {
      setResetStatus(`Error: ${error instanceof Error ? error.message : "Unknown"}`);
      setTimeout(() => setResetStatus(null), 5000);
    }
  };

  return (
    <article className={`panel ear-debug-panel ${className}`}>
      <div className="panel-header">
        <h3>🔍 Blink Detection Debug</h3>
        <button
          className="ghost-button reset-button"
          onClick={handleReset}
          title="Clear liveness cache - users will need to blink again"
        >
          Reset Liveness
        </button>
      </div>

      {resetStatus && (
        <div className="reset-status">{resetStatus}</div>
      )}

      <div className="ear-stats">
        <div className="stat-row">
          <span className="stat-label">Current EAR:</span>
          <span className={`stat-value ${currentEar != null && currentEar < threshold ? "closed" : "open"}`}>
            {currentEar?.toFixed(3) ?? "--"}
          </span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Min / Max:</span>
          <span className="stat-value">
            {minEar?.toFixed(3) ?? "--"} / {maxEar?.toFixed(3) ?? "--"}
          </span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Threshold:</span>
          <span className="stat-value threshold">{threshold}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Frames:</span>
          <span className="stat-value">{framesProcessed}</span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Status:</span>
          <span className={`stat-value blink-status ${hasBlinked ? "passed" : blinksNeeded > 0 ? "waiting" : "unknown"}`}>
            {hasBlinked ? "✓ Blink Detected" : blinksNeeded > 0 ? "⏳ Waiting for blink..." : "Unknown"}
          </span>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        width={280}
        height={100}
        className="ear-graph"
      />

      <div className="ear-legend">
        <span className="legend-item">
          <span className="legend-dot open"></span> Eyes Open
        </span>
        <span className="legend-item">
          <span className="legend-dot closed"></span> Eyes Closed
        </span>
      </div>
    </article>
  );
}
