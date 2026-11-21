from __future__ import annotations

from pathlib import Path

import pytest

SEQINFO = """[Sequence]
name=MOT20-TINY
imDir=img1
frameRate=25
seqLength=3
imWidth=1920
imHeight=1080
imExt=.jpg
"""

GT = """1,1,100,200,40,120,1,1,0.9
1,2,300,220,42,118,1,7,1.0
2,1,102,201,40,120,1,1,0.85
3,1,104,202,40,120,1,1,0.8
"""


@pytest.fixture
def tiny_seq(tmp_path: Path) -> Path:
    """A 3-frame MOTChallenge sequence with placeholder (empty) image files."""
    seq = tmp_path / "MOT20-TINY"
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    (seq / "seqinfo.ini").write_text(SEQINFO)
    (seq / "gt" / "gt.txt").write_text(GT)
    for i in range(1, 4):
        (seq / "img1" / f"{i:06d}.jpg").touch()
    return seq
