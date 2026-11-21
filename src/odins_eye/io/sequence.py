"""Stream frames of a MOTChallenge sequence one at a time.

Frames are read lazily so memory stays constant regardless of sequence length
(a 1080p MOT20 sequence held fully in memory can exceed 15 GB).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from odins_eye.io.mot import SeqInfo, read_seqinfo


def iter_frame_paths(
    seq_dir: str | Path,
    start: int = 1,
    stop: int | None = None,
) -> Iterator[tuple[int, Path]]:
    """Yield ``(frame_index, image_path)`` for frames ``start..stop`` (1-indexed, inclusive)."""
    seq_dir = Path(seq_dir)
    info: SeqInfo = read_seqinfo(seq_dir)
    last = info.seq_length if stop is None else min(stop, info.seq_length)
    if start < 1:
        raise ValueError("start must be >= 1 (frames are 1-indexed)")

    img_dir = seq_dir / info.im_dir
    for idx in range(start, last + 1):
        path = img_dir / f"{idx:06d}{info.im_ext}"
        if not path.is_file():
            raise FileNotFoundError(f"missing frame {idx}: {path}")
        yield idx, path


def iter_frames(
    seq_dir: str | Path,
    start: int = 1,
    stop: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR image)``, decoding one frame at a time.

    Requires the ``video`` extra (``pip install -e ".[video]"``).
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - exercised only without OpenCV
        raise ImportError('iter_frames needs OpenCV: pip install -e ".[video]"') from e

    for idx, path in iter_frame_paths(seq_dir, start, stop):
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"could not decode frame {idx}: {path}")
        yield idx, img
