#!/usr/bin/env python3
"""
Enrollment script: build or update the face embeddings database from known-person images.

Input: data/known/ where each subfolder name is the person label.
Output: data/encodings/encodings.pkl containing labels, embeddings, and metadata.

When multiple faces are found in an image, the largest face is used.
Supports incremental update: loads existing encodings and appends new ones, avoiding duplicates
by (image path, file modified time).
"""

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import face_recognition
import numpy as np

from config import (
    DEFAULT_ENCODINGS_PATH,
    DEFAULT_KNOWN_DIR,
    ENCODINGS_VERSION,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from utils import get_logger, load_image_rgb, list_image_paths

logger = get_logger(__name__)

# Database structure: dict with keys "labels", "encodings", "metadata"
# metadata: list of dicts with "image_path", "timestamp", "model", "version", "source_hash"


def _source_hash(image_path: str) -> str:
    """Compute a hash for duplicate detection: path + mtime."""
    try:
        mtime = os.path.getmtime(image_path)
        s = f"{os.path.abspath(image_path)}:{mtime}"
        return hashlib.sha256(s.encode()).hexdigest()
    except OSError:
        return hashlib.sha256(image_path.encode()).hexdigest()


def _load_existing_db(encodings_path: str) -> Tuple[List[str], List[np.ndarray], List[Dict[str, Any]]]:
    """Load existing encodings file. Returns (labels, encodings, metadata)."""
    if not os.path.isfile(encodings_path):
        return [], [], []
    try:
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        labels = data.get("labels", [])
        encodings = data.get("encodings", [])
        metadata = data.get("metadata", [])
        if len(metadata) < len(labels):
            metadata = metadata + [{}] * (len(labels) - len(metadata))
        logger.info("Loaded %d existing encodings from %s", len(labels), encodings_path)
        return labels, encodings, metadata
    except Exception as e:
        logger.warning("Could not load existing encodings: %s. Starting fresh.", e)
        return [], [], []


def _save_db(
    encodings_path: str,
    labels: List[str],
    encodings: List[np.ndarray],
    metadata: List[Dict[str, Any]],
) -> None:
    """Save encodings database to disk."""
    os.makedirs(os.path.dirname(encodings_path) or ".", exist_ok=True)
    data = {
        "labels": labels,
        "encodings": encodings,
        "metadata": metadata,
        "version": ENCODINGS_VERSION,
        "model": "face_recognition (dlib)",
    }
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved %d encodings to %s", len(labels), encodings_path)


def _process_image(
    image_path: str,
    label: str,
    model: str,
    num_jitters: int,
    existing_hashes: Optional[set] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Load image, detect one face (largest if multiple), compute embedding.
    Returns (encoding, metadata_dict) or None if skipped.
    """
    existing_hashes = existing_hashes or set()
    source_hash = _source_hash(image_path)
    if source_hash in existing_hashes:
        logger.debug("Skipping duplicate source: %s", image_path)
        return None

    rgb = load_image_rgb(image_path)
    if rgb is None:
        logger.warning("Could not load image: %s", image_path)
        return None

    # face_recognition uses (top, right, bottom, left)
    locations = face_recognition.face_locations(rgb, model=model)
    if not locations:
        logger.warning("No face found in %s", image_path)
        return None
    if len(locations) > 1:
        # Use largest face by area
        def area(loc: Tuple[int, int, int, int]) -> int:
            t, r, b, l = loc
            return (b - t) * (r - l)
        locations = [max(locations, key=area)]
        logger.debug("Multiple faces in %s; using largest", image_path)

    encodings = face_recognition.face_encodings(rgb, known_face_locations=locations, num_jitters=num_jitters)
    if not encodings:
        logger.warning("Could not compute encoding for %s", image_path)
        return None

    import time
    meta = {
        "image_path": image_path,
        "timestamp": time.time(),
        "model": model,
        "version": ENCODINGS_VERSION,
        "source_hash": source_hash,
    }
    return (encodings[0], meta)


def enroll(
    known_dir: str,
    encodings_path: str,
    detection_model: str = "hog",
    num_jitters: int = 1,
    incremental: bool = True,
) -> int:
    """
    Scan known_dir for person subfolders, compute face encodings, and save/update encodings_path.
    Returns the total number of encodings saved.
    """
    known_path = Path(known_dir)
    if not known_path.is_dir():
        logger.error("Known directory does not exist: %s", known_dir)
        return 0

    # Subfolders = person labels
    person_dirs = [d for d in known_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not person_dirs:
        logger.warning("No person subfolders found in %s", known_dir)
        # Still try to load existing and save (so we don't wipe the DB)
        if incremental:
            labels, encodings, metadata = _load_existing_db(encodings_path)
            if labels:
                _save_db(encodings_path, labels, encodings, metadata)
        return 0

    existing_hashes: set = set()
    if incremental:
        existing_labels, existing_encodings, existing_metadata = _load_existing_db(encodings_path)
        existing_hashes = {m.get("source_hash") for m in existing_metadata if m.get("source_hash")}
        all_labels = list(existing_labels)
        all_encodings = list(existing_encodings)
        all_metadata = list(existing_metadata)
    else:
        all_labels = []
        all_encodings = []
        all_metadata = []

    extensions = tuple(ext.lower() for ext in SUPPORTED_IMAGE_EXTENSIONS)
    added = 0

    for person_dir in sorted(person_dirs):
        label = person_dir.name
        image_paths = list_image_paths(str(person_dir), extensions)
        for image_path in image_paths:
            result = _process_image(
                image_path,
                label,
                detection_model,
                num_jitters,
                existing_hashes,
            )
            if result is None:
                continue
            encoding, meta = result
            existing_hashes.add(meta["source_hash"])
            all_labels.append(label)
            all_encodings.append(encoding)
            all_metadata.append(meta)
            added += 1
            logger.info("Enrolled: %s <- %s", label, image_path)

    if not all_labels and not incremental:
        logger.warning("No encodings produced. Not writing empty file.")
        return 0

    _save_db(encodings_path, all_labels, all_encodings, all_metadata)
    return len(all_labels)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or update face encodings database from known-person images.",
    )
    parser.add_argument(
        "--known_dir",
        default=DEFAULT_KNOWN_DIR,
        help="Root folder where each subfolder is a person label (default: data/known)",
    )
    parser.add_argument(
        "--encodings_path",
        default=DEFAULT_ENCODINGS_PATH,
        help="Output encodings file (default: data/encodings/encodings.pkl)",
    )
    parser.add_argument(
        "--detection_model",
        choices=("hog", "cnn"),
        default="hog",
        help="Face detection model: hog (faster) or cnn (default: hog)",
    )
    parser.add_argument(
        "--num_jitters",
        type=int,
        default=1,
        help="Number of jitters for face_encodings (default: 1)",
    )
    parser.add_argument(
        "--no_incremental",
        action="store_true",
        help="Ignore existing encodings and build from scratch",
    )
    args = parser.parse_args()

    from utils import setup_logging
    setup_logging()

    count = enroll(
        known_dir=args.known_dir,
        encodings_path=args.encodings_path,
        detection_model=args.detection_model,
        num_jitters=args.num_jitters,
        incremental=not args.no_incremental,
    )
    logger.info("Enrollment complete. Total encodings: %d", count)
    return 0 if count >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
