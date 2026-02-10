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
    DEFAULT_AUTHORIZED_NAME,
    DEFAULT_VAULT_TITLE,
    DEFAULT_VAULT_OPEN_HOLD_SEC,
    DEFAULT_VAULT_HYSTERESIS_FRAMES,
    DEFAULT_VAULT_OVERLAY_TEXT,
    DEFAULT_VAULT_WINDOW_WIDTH,
    DEFAULT_VAULT_WINDOW_HEIGHT,
    VAULT_EVENTS_LOG_DIR,
    COLOR_KNOWN,
    COLOR_UNKNOWN,
    LABEL_FONT_SCALE,
    LABEL_THICKNESS,
    RECOGNITION_LOG_COOLDOWN_SECONDS,
    DEFAULT_SNAPSHOT_DIR,
)
from utils import (
    draw_face_box,
    get_logger,
    setup_logging,
    render_vault_frame,
    update_vault_state,
    log_vault_event,
)

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
    authorized_name: str,
    vault_title: str,
    vault_open_hold_sec: float,
    vault_hysteresis_frames: int,
    vault_overlay_text: bool,
    vault_window_width: int,
    vault_window_height: int,
    no_camera: bool = False,
    test_frames_dir: Optional[str] = None,
) -> int:
    """Main loop: load encodings, open camera (or test frames), process frames, draw, vault window, handle keys."""
    known_labels, known_encodings, has_encodings = load_encodings(encodings_path)
    if not has_encodings:
        logger.warning("No known identities. Running in detection-only mode (all faces labeled Unknown).")

    cap: Optional[cv2.VideoCapture] = None
    test_frames: List[np.ndarray] = []
    if no_camera and test_frames_dir and os.path.isdir(test_frames_dir):
        from utils import list_image_paths
        from config import SUPPORTED_IMAGE_EXTENSIONS
        exts = tuple(ext.lower() for ext in SUPPORTED_IMAGE_EXTENSIONS)
        paths = list_image_paths(test_frames_dir, exts)
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                test_frames.append(img)
        if not test_frames:
            logger.warning("No images in %s; using a blank frame for simulation.", test_frames_dir)
    elif no_camera:
        test_frames = []  # will use a single synthetic frame per iteration

    if cap is None and not no_camera:
        cap = open_camera(camera_index, frame_width, frame_height)
        if cap is None:
            return 1
    elif no_camera and not test_frames:
        cap = None  # will generate a dummy frame below

    window_name = "Camera – Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(vault_title, cv2.WINDOW_NORMAL)

    frame_count = 0
    last_boxes: List[Tuple[int, int, int, int, str, float]] = []  # top, right, bottom, left, name, distance
    last_log_time: Dict[str, float] = {}
    snapshot_count = 0

    # Vault state (anti-flicker: hysteresis + hold timer)
    vault_state = "locked"
    last_authorized_time = 0.0
    authorized_confirm_count = 0
    last_authorized_distance: Optional[float] = None  # for VAULT_OPEN log
    last_authorized_detected = False  # only valid on recognition frames; used when calling update_vault_state

    def get_frame() -> Tuple[bool, Optional[np.ndarray]]:
        if cap is not None:
            ret, frame = cap.read()
            return ret, frame
        if test_frames:
            idx = frame_count % len(test_frames)
            return True, test_frames[idx].copy()
        # Single synthetic frame (no_camera, no test_frames_dir or empty)
        h, w = frame_height, frame_width
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (60, 60, 60)
        cv2.putText(frame, "No camera / test frames", (w // 4, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return True, frame

    try:
        while True:
            ret, frame = get_frame()
            if not ret or frame is None:
                logger.debug("Failed to read frame")
                time.sleep(0.05)
                continue

            frame_count += 1
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            scale_x = frame.shape[1] / rgb_small.shape[1]
            scale_y = frame.shape[0] / rgb_small.shape[0]

            is_recognition_frame = (frame_count % process_every_n_frames == 0)
            authorized_detected = False
            best_authorized_distance: Optional[float] = None

            if is_recognition_frame:
                locations = face_recognition.face_locations(rgb_small, model=detection_model)
                last_boxes = []
                for (top, right, bottom, left) in locations:
                    top_f = int(top * scale_y)
                    right_f = int(right * scale_x)
                    bottom_f = int(bottom * scale_y)
                    left_f = int(left * scale_x)
                    encodings = face_recognition.face_encodings(rgb_small, known_face_locations=[(top, right, bottom, left)])
                    if not encodings:
                        last_boxes.append((top_f, right_f, bottom_f, left_f, "Unknown", float("inf")))
                        continue
                    name, dist = recognize_face(encodings[0], known_labels, known_encodings, threshold)
                    last_boxes.append((top_f, right_f, bottom_f, left_f, name, dist))
                    if name != "Unknown":
                        now = time.time()
                        if name not in last_log_time or (now - last_log_time[name]) >= RECOGNITION_LOG_COOLDOWN_SECONDS:
                            last_log_time[name] = now
                            logger.info("Recognized: %s (distance=%.3f, frame=%d)", name, dist, frame_count)
                # Authorized condition: any face with exact name match and distance <= threshold
                authorized_matches = [(name, dist) for (_, _, _, _, name, dist) in last_boxes if name == authorized_name and dist <= threshold]
                authorized_detected = len(authorized_matches) > 0
                if authorized_matches:
                    best_authorized_distance = min(d for _, d in authorized_matches)
                last_authorized_detected = authorized_detected

            now = time.time()
            new_state, debug = update_vault_state(
                vault_state,
                last_authorized_detected if is_recognition_frame else False,
                last_authorized_time,
                authorized_confirm_count,
                now,
                vault_open_hold_sec,
                vault_hysteresis_frames,
                is_recognition_frame=is_recognition_frame,
            )
            transition = debug.get("transition")
            if transition == "locked -> open":
                last_authorized_distance = best_authorized_distance if best_authorized_distance is not None else debug.get("distance")
                log_vault_event(
                    "VAULT_OPEN",
                    authorized_name=authorized_name,
                    distance=last_authorized_distance,
                    camera_frame_id=frame_count,
                    log_dir=VAULT_EVENTS_LOG_DIR,
                )
                logger.info("Vault OPEN (authorized_name=%s, distance=%.3f, frame=%d)", authorized_name, last_authorized_distance or 0, frame_count)
            elif transition == "open -> locked":
                log_vault_event("VAULT_LOCK", authorized_name=authorized_name, reason="timeout", log_dir=VAULT_EVENTS_LOG_DIR)
                logger.info("Vault LOCKED (reason=timeout)")

            vault_state = new_state
            authorized_confirm_count = int(debug.get("new_confirm_count", authorized_confirm_count))
            last_authorized_time = float(debug.get("new_last_authorized_time", last_authorized_time))

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

            vault_frame = render_vault_frame(
                vault_state == "open",
                vault_title,
                vault_window_width,
                vault_window_height,
                show_text=vault_overlay_text,
            )
            cv2.imshow(vault_title, vault_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
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
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0


def _parse_vault_size(s: str) -> Tuple[int, int]:
    """Parse 'WxH' into (width, height)."""
    parts = s.strip().lower().split("x")
    if len(parts) != 2:
        raise ValueError("vault_window_size must be WxH, e.g. 640x480")
    return int(parts[0]), int(parts[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-time face detection and recognition from webcam with TREZOR vault access control demo.",
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
    # Vault (TREZOR) access control demo
    parser.add_argument("--authorized_name", default=DEFAULT_AUTHORIZED_NAME, help="Identity that can open the vault (default: Queen Elizabeth)")
    parser.add_argument("--vault_title", default=DEFAULT_VAULT_TITLE, help="Vault window title (default: TREZOR)")
    parser.add_argument("--vault_open_hold_sec", type=float, default=DEFAULT_VAULT_OPEN_HOLD_SEC, help="Seconds to keep vault open after last authorized detection (default: 3.0)")
    parser.add_argument("--vault_hysteresis_frames", type=int, default=DEFAULT_VAULT_HYSTERESIS_FRAMES, help="Frames with authorized match required before opening (default: 2)")
    parser.add_argument("--vault_overlay_text", action="store_true", default=DEFAULT_VAULT_OVERLAY_TEXT, help="Show LOCKED/OPEN and ACCESS DENIED/GRANTED text (default: True)")
    parser.add_argument("--no_vault_overlay_text", action="store_false", dest="vault_overlay_text", help="Disable vault overlay text")
    parser.add_argument("--vault_window_size", type=str, default=f"{DEFAULT_VAULT_WINDOW_WIDTH}x{DEFAULT_VAULT_WINDOW_HEIGHT}", help="Vault window size WxH (default: 640x480)")
    parser.add_argument("--no_camera", action="store_true", help="Simulation mode: no webcam; use frames from --test_frames_dir or a placeholder")
    parser.add_argument("--test_frames_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test", "video_frames"), help="Folder of images for --no_camera simulation")
    args = parser.parse_args()

    try:
        vault_w, vault_h = _parse_vault_size(args.vault_window_size)
    except ValueError as e:
        parser.error(str(e))

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
        authorized_name=args.authorized_name,
        vault_title=args.vault_title,
        vault_open_hold_sec=args.vault_open_hold_sec,
        vault_hysteresis_frames=args.vault_hysteresis_frames,
        vault_overlay_text=args.vault_overlay_text,
        vault_window_width=vault_w,
        vault_window_height=vault_h,
        no_camera=args.no_camera,
        test_frames_dir=args.test_frames_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
