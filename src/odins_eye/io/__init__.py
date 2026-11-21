"""Input/output: MOTChallenge sequences, annotations and result files."""

from odins_eye.io.mot import (
    MOT20_DISTRACTOR_CLASSES,
    PEDESTRIAN_CLASS,
    SeqInfo,
    read_mot_txt,
    read_seqinfo,
    write_mot_results,
)
from odins_eye.io.sequence import iter_frame_paths, iter_frames

__all__ = [
    "MOT20_DISTRACTOR_CLASSES",
    "PEDESTRIAN_CLASS",
    "SeqInfo",
    "iter_frame_paths",
    "iter_frames",
    "read_mot_txt",
    "read_seqinfo",
    "write_mot_results",
]
