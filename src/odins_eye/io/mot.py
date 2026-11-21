"""Read and write MOTChallenge files.

Conventions (see CONTRIBUTING.md):
  * frames are 1-indexed, as in MOTChallenge;
  * the frame rate is always read from ``seqinfo.ini``, never assumed;
  * boxes are ``(x, y, w, h)`` in pixels, top-left origin.

Ground-truth rows have 9 columns::

    frame, id, x, y, w, h, consider, class, visibility

Result rows have 10 columns (the last three are unused and set to -1)::

    frame, id, x, y, w, h, conf, -1, -1, -1
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PEDESTRIAN_CLASS = 1

# Classes whose matched predictions the official MOT20 evaluation ignores instead of
# counting as false positives: person on vehicle (2), non-motorized vehicle (6),
# static person (7), distractor (8) and reflection (12).
MOT20_DISTRACTOR_CLASSES: frozenset[int] = frozenset({2, 6, 7, 8, 12})


@dataclass(frozen=True)
class SeqInfo:
    """Metadata of one MOTChallenge sequence, parsed from ``seqinfo.ini``."""

    name: str
    im_dir: str
    frame_rate: float
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str


def read_seqinfo(path: str | Path) -> SeqInfo:
    """Parse a ``seqinfo.ini`` file, or the one inside a sequence directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "seqinfo.ini"
    if not path.is_file():
        raise FileNotFoundError(f"seqinfo.ini not found: {path}")

    parser = configparser.ConfigParser()
    parser.read(path)
    if "Sequence" not in parser:
        raise ValueError(f"{path} has no [Sequence] section")
    s = parser["Sequence"]
    return SeqInfo(
        name=s["name"],
        im_dir=s.get("imDir", "img1"),
        frame_rate=s.getfloat("frameRate"),
        seq_length=s.getint("seqLength"),
        im_width=s.getint("imWidth"),
        im_height=s.getint("imHeight"),
        im_ext=s.get("imExt", ".jpg"),
    )


def read_mot_txt(path: str | Path, min_cols: int = 6) -> np.ndarray:
    """Load a MOTChallenge text file into a float64 array of shape ``(N, C)``.

    An empty file returns an array of shape ``(0, min_cols)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MOT file not found: {path}")
    if path.stat().st_size == 0:
        return np.zeros((0, min_cols), dtype=np.float64)

    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if data.shape[1] < min_cols:
        raise ValueError(f"{path}: expected at least {min_cols} columns, got {data.shape[1]}")
    if np.any(data[:, 0] < 1):
        raise ValueError(f"{path}: frame indices must be 1-indexed")
    return data


def write_mot_results(path: str | Path, rows: np.ndarray) -> None:
    """Write tracker output in MOTChallenge result format.

    ``rows`` has shape ``(N, 7)``: ``frame, id, x, y, w, h, conf``. Rows are sorted by
    frame, then id, which is what TrackEval expects.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"rows must have shape (N, 7), got {rows.shape}")
    if rows.size and np.any(rows[:, 0] < 1):
        raise ValueError("frame indices must be 1-indexed")
    if rows.size and np.any(rows[:, 4:6] <= 0):
        raise ValueError("box width and height must be positive")

    order = np.lexsort((rows[:, 1], rows[:, 0]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame, tid, x, y, w, h, conf in rows[order]:
            f.write(
                f"{int(frame)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
            )
