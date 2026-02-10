"""
Shared utilities: image loading, logging, and drawing helpers.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional

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
