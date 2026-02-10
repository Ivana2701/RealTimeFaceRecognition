#!/usr/bin/env python3
"""
Real-time face detection and recognition from webcam.

Loads encodings from disk, opens camera, detects faces, matches to known identities,
and draws bounding boxes with labels. Supports hot-reload of encodings ('r') and
snapshot ('s'). Runs in detection-only mode if no encodings are available.
"""

import argparse
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_DETECTION_MODEL,
    DEFAULT_ENCODINGS_PATH,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_KNOWN_DIR,
    DEFAULT_PROCESS_EVERY_N_FRAMES,
    DEFAULT_THRESHOLD,
    COLOR_KNOWN,
    COLOR_UNKNOWN,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    RECOGNITION_LOG_COOLDOWN_SECONDS,
    DEFAULT_SNAPSHOT_DIR,
)
from utils import draw_face_box, get_logger, setup_logging

logger = get_logger(__name__)


def load_encodings(encodings_path: str) -> Tuple[List[str], List[np.ndarray], bool]:
    """
    Load encodings database from disk.
    Returns (labels, encodings, success). On failure returns ([], [], False).
    """
    if not os.path.isfile(encodings_path):
        logger.warning("Encodings file not found: %s", encodings_path)
        return [], [], False
    try:
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        labels = data.get("labels", [])
        encodings = data.get("encodings", [])
        if not labels or not encodings or len(labels) != len(encodings):
            logger.warning("Invalid encodings file: empty or length mismatch")
            return [], [], False
        logger.info("Loaded %d known identities from %s", len(set(labels)), encodings_path)
        return labels, encodings, True
    except Exception as e:
        logger.exception("Failed to load encodings: %s", e)
        return [], [], False


def recognize_face(
    encoding: np.ndarray,
    known_labels: List[str],
    known_encodings: List[np.ndarray],
    threshold: float,
) -> Tuple[str, float]:
    """
    Compare a face encoding to known encodings. Returns (name, distance).
    name is "Unknown" if best distance > threshold.
    """
    if not known_encodings:
        return "Unknown", float("inf")
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    if best_distance <= threshold:
        return known_labels[best_idx], best_distance
    return "Unknown", best_distance


def open_camera(camera_index: int, width: int, height: int) -> Optional[cv2.VideoCapture]:
    """Open webcam and set resolution. Returns VideoCapture or None on failure."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run(
    encodings_path: str,
    camera_index: int,
    frame_width: int,
    frame_height: int,
    process_every_n_frames: int,
    threshold: float,
    detection_model: str,
    snapshot_dir: str,
) -> int:
    """Main loop: load encodings, open camera, process frames, draw, handle keys."""
    known_labels, known_encodings, has_encodings = load_encodings(encodings_path)
    if not has_encodings:
        logger.warning("No known identities. Running in detection-only mode (all faces labeled Unknown).")

    cap = open_camera(camera_index, frame_width, frame_height)
    if cap is None:
        return 1

    window_name = "Face Recognition (q/ESC quit, r reload, s snapshot)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    last_boxes: List[Tuple[int, int, int, int, str, float]] = []  # top, right, bottom, left, name, distance
    last_log_time: Dict[str, float] = {}
    snapshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug("Failed to read frame")
                time.sleep(0.05)
                continue

            frame_count += 1
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            scale_x = frame.shape[1] / rgb_small.shape[1]
            scale_y = frame.shape[0] / rgb_small.shape[0]

            # Run recognition every N frames
            if frame_count % process_every_n_frames == 0:
                locations = face_recognition.face_locations(rgb_small, model=detection_model)
                last_boxes = []
                for (top, right, bottom, left) in locations:
                    # Scale back to full frame coordinates
                    top_f = int(top * scale_y)
                    right_f = int(right * scale_x)
                    bottom_f = int(bottom * scale_y)
                    left_f = int(left * scale_x)
                    # Encoding from small frame; face_locations already in small frame
                    encodings = face_recognition.face_encodings(rgb_small, known_face_locations=[(top, right, bottom, left)])
                    if not encodings:
                        last_boxes.append((top_f, right_f, bottom_f, left_f, "Unknown", float("inf")))
                        continue
                    name, dist = recognize_face(encodings[0], known_labels, known_encodings, threshold)
                    last_boxes.append((top_f, right_f, bottom_f, left_f, name, dist))
                    # Log with cooldown
                    if name != "Unknown":
                        now = time.time()
                        if name not in last_log_time or (now - last_log_time[name]) >= RECOGNITION_LOG_COOLDOWN_SECONDS:
                            last_log_time[name] = now
                            logger.info("Recognized: %s (distance=%.3f, frame=%d)", name, dist, frame_count)

            # Draw last known boxes on current frame
            for (top, right, bottom, left, name, dist) in last_boxes:
                color = COLOR_KNOWN if name != "Unknown" else COLOR_UNKNOWN
                sublabel = f"{dist:.2f}" if name != "Unknown" else None
                draw_face_box(
                    frame,
                    top, right, bottom, left,
                    name,
                    color,
                    sublabel=sublabel,
                    font_scale=LABEL_FONT_SCALE,
                    thickness=LABEL_THICKNESS,
                )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            if key == ord("r"):
                new_labels, new_encodings, ok = load_encodings(encodings_path)
                if ok:
                    known_labels, known_encodings = new_labels, new_encodings
                    logger.info("Reloaded encodings (%d identities)", len(set(known_labels)))
                else:
                    logger.warning("Reload failed; keeping previous encodings")
            if key == ord("s"):
                os.makedirs(snapshot_dir, exist_ok=True)
                snapshot_count += 1
                path = os.path.join(snapshot_dir, f"snapshot_{snapshot_count:04d}.jpg")
                cv2.imwrite(path, frame)
                logger.info("Saved snapshot: %s", path)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-time face detection and recognition from webcam.",
    )
    parser.add_argument("--known_dir", default=DEFAULT_KNOWN_DIR, help="Known images root (for message only)")
    parser.add_argument("--encodings_path", default=DEFAULT_ENCODINGS_PATH, help="Encodings file (default: data/encodings/encodings.pkl)")
    parser.add_argument("--camera_index", type=int, default=DEFAULT_CAMERA_INDEX, help="Webcam index (default: 0)")
    parser.add_argument("--frame_width", type=int, default=DEFAULT_FRAME_WIDTH, help="Frame width (default: 640)")
    parser.add_argument("--frame_height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Frame height (default: 480)")
    parser.add_argument("--process_every_n_frames", type=int, default=DEFAULT_PROCESS_EVERY_N_FRAMES, help="Run recognition every N frames (default: 3)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Match threshold; lower=stricter (default: 0.6)")
    parser.add_argument("--detection_model", choices=("hog", "cnn"), default=DEFAULT_DETECTION_MODEL, help="Detection model (default: hog)")
    parser.add_argument("--snapshot_dir", default=DEFAULT_SNAPSHOT_DIR, help="Directory for snapshots (default: data/snapshots)")
    args = parser.parse_args()

    setup_logging()
    return run(
        encodings_path=args.encodings_path,
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        process_every_n_frames=args.process_every_n_frames,
        threshold=args.threshold,
        detection_model=args.detection_model,
        snapshot_dir=args.snapshot_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
