"""UI overlay helpers to keep GUI logic lean."""
import cv2
from typing import Tuple

from src.pipeline.face_processor import draw_rounded_rectangle


def draw_ear_graph(frame, ear_history, ear_threshold: float = 0.5):
    """Draw EAR (Eye Aspect Ratio) graph overlay for blink debugging."""
    if len(ear_history) < 1:
        return frame

    h, w = frame.shape[:2]
    graph_x, graph_y, graph_w, graph_h = 10, h - 130, 300, 120

    if len(ear_history) < 2:
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (100, 100, 100), 2)
        cv2.putText(
            frame,
            "EAR Graph - Collecting data...",
            (graph_x + 5, graph_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        return frame

    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (100, 100, 100), 2)

    threshold_y = int(graph_y + graph_h - (ear_threshold / 0.8 * graph_h))
    cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"Threshold: {ear_threshold:.2f}",
        (graph_x + 5, threshold_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 255),
        1,
    )

    for i in range(1, len(ear_history)):
        x1 = graph_x + int((i - 1) * graph_w / 150)
        x2 = graph_x + int(i * graph_w / 150)
        y1 = graph_y + graph_h - int(min(ear_history[i - 1], 0.8) / 0.8 * graph_h)
        y2 = graph_y + graph_h - int(min(ear_history[i], 0.8) / 0.8 * graph_h)
        line_color = (0, 255, 0) if ear_history[i] > ear_threshold else (0, 0, 255)
        cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

    current_ear = ear_history[-1]
    ear_color = (0, 255, 0) if current_ear > ear_threshold else (0, 0, 255)
    cv2.putText(frame, "EAR Over Time", (graph_x + 5, graph_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Current: {current_ear:.3f}", (graph_x + 5, graph_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ear_color, 1)
    return frame


def draw_registration_overlay(
    frame,
    instruction_text: str,
    current_step: int,
    total_steps: int,
    feedback_text: str = "",
    feedback_color: Tuple[int, int, int] = (255, 165, 0),
):
    """Render the registration instruction card and progress on the frame."""
    if total_steps <= 0:
        return frame

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX

    (inst_w, inst_h), _ = cv2.getTextSize(instruction_text, font, 0.9, 2)
    card_w = min(w - 100, 500)
    card_h = 80
    x_offset = (w - card_w) // 2
    y_offset = 20

    overlay = frame.copy()
    draw_rounded_rectangle(overlay, (x_offset, y_offset), (x_offset + card_w, y_offset + card_h), (26, 26, 46), -1, radius=12)
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

    progress = max(0.0, min(1.0, current_step / total_steps))
    circle_x = x_offset + 50
    circle_y = y_offset + card_h // 2
    radius = 28

    cv2.circle(frame, (circle_x, circle_y), radius, (60, 60, 80), -1)
    angle = int(360 * progress)
    cv2.ellipse(frame, (circle_x, circle_y), (radius - 3, radius - 3), -90, 0, angle, (46, 204, 113), 4)
    progress_text = f"{current_step + 1}/{total_steps}"
    (prog_w, prog_h), _ = cv2.getTextSize(progress_text, font, 0.6, 1)
    cv2.putText(frame, progress_text, (circle_x - prog_w // 2, circle_y + prog_h // 2), font, 0.6, (255, 255, 255), 1)

    text_x = x_offset + 100
    text_y = y_offset + (card_h + inst_h) // 2
    cv2.putText(frame, instruction_text, (text_x, text_y), font, 0.85, (255, 255, 255), 1, cv2.LINE_AA)

    if feedback_text:
        cv2.putText(frame, feedback_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)

    return frame


def draw_status_chip(frame, text: str, color: Tuple[int, int, int], progress_pct: float | None = None):
    """Draw a top-right status chip with optional progress bar."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)

    x_pos = w - tw - 40
    y_pos = 10

    overlay = frame.copy()
    draw_rounded_rectangle(overlay, (x_pos - 10, y_pos), (x_pos + tw + 20, y_pos + th + 16), color, -1, radius=10)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, text, (x_pos + 5, y_pos + th + 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if progress_pct is not None:
        bar_x = x_pos - 10
        bar_y = y_pos + th + 20
        bar_w = tw + 30
        bar_h = 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 80), -1)
        progress_w = int(bar_w * max(0.0, min(1.0, progress_pct / 100.0)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + bar_h), color, -1)

    return frame


def draw_banner(frame, text: str, bg_color: Tuple[int, int, int], alpha: float = 0.85, radius: int = 8):
    """Draw a top-left banner with rounded background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
    overlay = frame.copy()
    draw_rounded_rectangle(overlay, (5, 5), (tw + 30, th + 22), bg_color, -1, radius=radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (18, th + 14), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
