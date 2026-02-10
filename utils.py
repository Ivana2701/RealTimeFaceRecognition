"""
Shared utilities: image loading, logging, drawing helpers, and vault (TREZOR) demo.
"""

import csv
import os
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load an image from disk. Returns BGR numpy array or None on failure.
    """
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return img


def load_image_rgb(path: str) -> Optional[np.ndarray]:
    """
    Load an image as RGB (required by face_recognition). Returns None on failure.
    """
    img = load_image(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_image_paths(root_dir: str, extensions: Tuple[str, ...]) -> List[str]:
    """
    Recursively list image file paths under root_dir with given extensions.
    """
    paths: List[str] = []
    root = Path(root_dir)
    if not root.is_dir():
        return paths
    for ext in extensions:
        paths.extend(str(p) for p in root.rglob(f"*{ext}"))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_face_box(
    frame: np.ndarray,
    top: int,
    right: int,
    bottom: int,
    left: int,
    label: str,
    color: Tuple[int, int, int],
    sublabel: Optional[str] = None,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """
    Draw a bounding box and label on the frame (in-place).
    top, right, bottom, left are in image coordinates.
    """
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Label background
    label_text = label
    if sublabel:
        label_text = f"{label} ({sublabel})"
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (left, top - th - 10), (left + tw + 4, top), color, -1)
    cv2.putText(
        frame,
        label_text,
        (left, top - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def rect_to_xyxy(top: int, right: int, bottom: int, left: int) -> Tuple[int, int, int, int]:
    """Convert (top, right, bottom, left) to (x1, y1, x2, y2)."""
    return (left, top, right, bottom)


def scale_rect(
    top: int, right: int, bottom: int, left: int,
    scale_x: float, scale_y: float,
) -> Tuple[int, int, int, int]:
    """Scale face box from downscaled frame coords back to original frame."""
    return (
        int(top * scale_y),
        int(right * scale_x),
        int(bottom * scale_y),
        int(left * scale_x),
    )


# ---------------------------------------------------------------------------
# Vault (TREZOR) access control demo – procedural rendering (no network assets)
# ---------------------------------------------------------------------------

# Gold palette (BGR): chest body, highlight, outline
VAULT_GOLD = (0, 180, 255)
VAULT_GOLD_LIGHT = (0, 220, 255)
VAULT_GOLD_DARK = (0, 120, 180)
VAULT_OUTLINE = (0, 0, 0)
VAULT_BG = (80, 80, 100)


def render_vault_frame(
    is_open: bool,
    title: str,
    w: int,
    h: int,
    show_text: bool = True,
) -> np.ndarray:
    """
    Render a procedural vault/chest visualization. No external assets.
    - CLOSED: gold chest with padlock, "LOCKED" and "ACCESS DENIED".
    - OPEN: chest with lid up, gold coins/bars inside, "OPEN" and "ACCESS GRANTED".
    Returns BGR image (HxWx3).
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = VAULT_BG

    cx, cy = w // 2, h // 2
    chest_w, chest_h = min(w, 400), min(h, 280)
    x1 = cx - chest_w // 2
    y1 = cy - chest_h // 2
    x2 = x1 + chest_w
    y2 = y1 + chest_h

    # Chest body (rounded rect effect via filled rect + outline)
    cv2.rectangle(frame, (x1, y1), (x2, y2), VAULT_GOLD_DARK, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), VAULT_OUTLINE, 3)
    # Highlight strip on top edge
    cv2.rectangle(frame, (x1 + 4, y1), (x2 - 4, y1 + 12), VAULT_GOLD_LIGHT, -1)
    cv2.rectangle(frame, (x1 + 4, y1), (x2 - 4, y1 + 12), VAULT_OUTLINE, 1)

    if is_open:
        # Lid "open" – drawn as a rectangle shifted up and tilted (simplified: just a rect above)
        lid_h = max(30, chest_h // 5)
        lid_y1 = y1 - lid_h - 5
        cv2.rectangle(frame, (x1, lid_y1), (x2, y1 + 5), VAULT_GOLD, -1)
        cv2.rectangle(frame, (x1, lid_y1), (x2, y1 + 5), VAULT_OUTLINE, 2)
        # Interior: gold coins (circles) and bars (rectangles)
        margin = 30
        in_x1, in_y1 = x1 + margin, y1 + 25
        in_x2, in_y2 = x2 - margin, y2 - 15
        # Coins
        for i in range(4):
            for j in range(3):
                cx_coin = in_x1 + (in_x2 - in_x1) * (i + 1) // 4
                cy_coin = in_y1 + (in_y2 - in_y1) * (j + 1) // 3
                cv2.circle(frame, (cx_coin, cy_coin), 18, VAULT_GOLD_LIGHT, -1)
                cv2.circle(frame, (cx_coin, cy_coin), 18, VAULT_OUTLINE, 1)
        # Bars (horizontal rectangles)
        bar_y = in_y2 - 35
        for i in range(3):
            bx = in_x1 + 25 + i * 55
            cv2.rectangle(frame, (bx, bar_y), (bx + 45, bar_y + 20), VAULT_GOLD, -1)
            cv2.rectangle(frame, (bx, bar_y), (bx + 45, bar_y + 20), VAULT_OUTLINE, 1)
    else:
        # Padlock (closed state)
        lock_w, lock_h = 50, 45
        lx = cx - lock_w // 2
        ly = cy - lock_h // 2 - 10
        # Lock body
        cv2.rectangle(frame, (lx, ly), (lx + lock_w, ly + lock_h), VAULT_GOLD_DARK, -1)
        cv2.rectangle(frame, (lx, ly), (lx + lock_w, ly + lock_h), VAULT_OUTLINE, 2)
        # Shackle (arch)
        cv2.ellipse(frame, (cx, ly - 5), (lock_w // 2 - 2, 18), 0, 180, 360, VAULT_OUTLINE, 3)
        cv2.rectangle(frame, (lx + 8, ly), (lx + 12, ly - 20), VAULT_OUTLINE, 2)
        cv2.rectangle(frame, (lx + lock_w - 12, ly), (lx + lock_w - 8, ly - 20), VAULT_OUTLINE, 2)

    if show_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        if is_open:
            status = f"{title}: OPEN"
            access = "ACCESS GRANTED"
            color = (0, 255, 0)  # Green
        else:
            status = f"{title}: LOCKED"
            access = "ACCESS DENIED"
            color = (0, 0, 255)  # Red
        (tw1, th1), _ = cv2.getTextSize(status, font, scale, thick)
        (tw2, th2), _ = cv2.getTextSize(access, font, scale, thick)
        cv2.putText(frame, status, (cx - tw1 // 2, 35), font, scale, color, thick, cv2.LINE_AA)
        cv2.putText(frame, access, (cx - tw2 // 2, h - 25), font, scale, color, thick, cv2.LINE_AA)

    return frame


def update_vault_state(
    current_state: str,
    authorized_detected: bool,
    last_authorized_time: float,
    authorized_confirm_count: int,
    now: float,
    vault_open_hold_sec: float,
    vault_hysteresis_frames: int,
    is_recognition_frame: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Vault state machine with hysteresis and hold timer.
    Call every frame; set is_recognition_frame=True only on frames where recognition ran.
    Returns (new_state, debug_info). debug_info has new_confirm_count, new_last_authorized_time for next call.
    State is "locked" or "open".
    """
    debug: Dict[str, Any] = {
        "authorized_detected": authorized_detected,
        "authorized_confirm_count": authorized_confirm_count,
        "last_authorized_time": last_authorized_time,
        "elapsed_since_authorized": now - last_authorized_time,
    }

    if is_recognition_frame:
        if authorized_detected:
            new_confirm = authorized_confirm_count + 1
            new_time = now
        else:
            new_confirm = 0
            new_time = last_authorized_time
        debug["new_confirm_count"] = new_confirm
        debug["new_last_authorized_time"] = new_time
    else:
        new_confirm = authorized_confirm_count
        new_time = last_authorized_time
        debug["new_confirm_count"] = new_confirm
        debug["new_last_authorized_time"] = new_time

    if current_state == "locked":
        if new_confirm >= vault_hysteresis_frames:
            new_state = "open"
            debug["transition"] = "locked -> open"
        else:
            new_state = "locked"
            debug["transition"] = None
        return new_state, debug

    # current_state == "open": check hold timer every frame
    if now - last_authorized_time <= vault_open_hold_sec:
        new_state = "open"
        debug["transition"] = None
        return new_state, debug
    else:
        new_state = "locked"
        debug["transition"] = "open -> locked"
        debug["new_confirm_count"] = 0
        return new_state, debug


_vault_log_initialized = False


def log_vault_event(
    event: str,
    authorized_name: str,
    reason: Optional[str] = None,
    distance: Optional[float] = None,
    camera_frame_id: Optional[int] = None,
    log_dir: Optional[str] = None,
    csv_filename: str = "vault_events.csv",
) -> None:
    """Append a vault event to CSV (outputs/logs/vault_events.csv)."""
    global _vault_log_initialized
    if log_dir is None:
        from config import VAULT_EVENTS_LOG_DIR
        log_dir = VAULT_EVENTS_LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, csv_filename)
    file_exists = os.path.isfile(path)
    try:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists or not _vault_log_initialized:
                writer.writerow(["timestamp", "event", "authorized_name", "reason", "distance", "camera_frame_id"])
                _vault_log_initialized = True
            row = [
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                event,
                authorized_name,
                reason or "",
                distance if distance is not None else "",
                camera_frame_id if camera_frame_id is not None else "",
            ]
            writer.writerow(row)
    except OSError as e:
        get_logger(__name__).warning("Could not write vault event to %s: %s", path, e)
