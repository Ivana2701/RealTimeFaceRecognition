"""
Default configuration for the face recognition application.
All values can be overridden via CLI arguments.
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KNOWN_DIR = os.path.join(PROJECT_ROOT, "data", "known")
DEFAULT_ENCODINGS_PATH = os.path.join(PROJECT_ROOT, "data", "encodings", "encodings.pkl")
DEFAULT_SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "data", "snapshots")

# Enrollment
ENCODINGS_VERSION = "1.0"
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Camera / real-time
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480

# Recognition
DEFAULT_THRESHOLD = 0.6  # face_recognition: lower = stricter, 0.55-0.65 typical
DEFAULT_PROCESS_EVERY_N_FRAMES = 3
DEFAULT_DETECTION_MODEL = "hog"  # "hog" (faster) or "cnn" (more accurate, GPU)
DEFAULT_NUM_JITTERS = 1  # face_encodings: more jitters = more accurate but slower

# Visualization
COLOR_KNOWN = (0, 255, 0)    # Green for recognized
COLOR_UNKNOWN = (0, 0, 255)  # Red for unknown
LABEL_FONT_SCALE = 0.6
LABEL_THICKNESS = 2

# Logging
RECOGNITION_LOG_COOLDOWN_SECONDS = 5.0  # Avoid spamming logs for same person

# Vault (TREZOR) access control demo
DEFAULT_AUTHORIZED_NAME = "Queen Elizabeth"
DEFAULT_VAULT_TITLE = "TREZOR"
DEFAULT_VAULT_OPEN_HOLD_SEC = 3.0  # Keep vault open after last authorized detection
DEFAULT_VAULT_HYSTERESIS_FRAMES = 2  # Require K consecutive authorized frames before opening
DEFAULT_VAULT_OVERLAY_TEXT = True
DEFAULT_VAULT_WINDOW_WIDTH = 640
DEFAULT_VAULT_WINDOW_HEIGHT = 480
VAULT_EVENTS_LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
VAULT_EVENTS_CSV = "vault_events.csv"
