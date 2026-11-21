from __future__ import annotations

import numpy as np
import pytest

from odins_eye.io import (
    MOT20_DISTRACTOR_CLASSES,
    PEDESTRIAN_CLASS,
    iter_frame_paths,
    read_mot_txt,
    read_seqinfo,
    write_mot_results,
)


def test_read_seqinfo_from_dir(tiny_seq):
    info = read_seqinfo(tiny_seq)
    assert info.name == "MOT20-TINY"
    assert info.frame_rate == 25.0
    assert info.seq_length == 3
    assert (info.im_width, info.im_height) == (1920, 1080)


def test_read_seqinfo_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_seqinfo(tmp_path)


def test_read_gt_and_split_classes(tiny_seq):
    gt = read_mot_txt(tiny_seq / "gt" / "gt.txt")
    assert gt.shape == (4, 9)
    classes = gt[:, 7].astype(int)
    assert (classes == PEDESTRIAN_CLASS).sum() == 3
    # The static person (class 7) must be treated as a distractor, not a false positive.
    assert set(classes[classes != PEDESTRIAN_CLASS]) <= MOT20_DISTRACTOR_CLASSES


def test_read_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.touch()
    assert read_mot_txt(p).shape == (0, 6)


def test_read_rejects_zero_indexed_frames(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0,1,10,10,5,5,1,-1,-1,-1\n")
    with pytest.raises(ValueError, match="1-indexed"):
        read_mot_txt(p)


def test_write_roundtrip_sorted(tmp_path):
    rows = np.array(
        [
            [2, 1, 10.0, 20.0, 30.0, 60.0, 0.9],
            [1, 2, 11.0, 21.0, 31.0, 61.0, 0.8],
            [1, 1, 12.0, 22.0, 32.0, 62.0, 0.7],
        ]
    )
    out = tmp_path / "res" / "MOT20-TINY.txt"
    write_mot_results(out, rows)

    back = read_mot_txt(out)
    assert back.shape == (3, 10)
    np.testing.assert_array_equal(back[:, :2], [[1, 1], [1, 2], [2, 1]])
    np.testing.assert_allclose(back[0, 2:7], [12.0, 22.0, 32.0, 62.0, 0.7])
    assert np.all(back[:, 7:] == -1)


@pytest.mark.parametrize(
    "rows, match",
    [
        (np.zeros((1, 6)), "shape"),
        (np.array([[0, 1, 0, 0, 5, 5, 1.0]]), "1-indexed"),
        (np.array([[1, 1, 0, 0, 0, 5, 1.0]]), "positive"),
    ],
)
def test_write_validates(tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        write_mot_results(tmp_path / "x.txt", rows)


def test_iter_frame_paths_streams_in_order(tiny_seq):
    frames = list(iter_frame_paths(tiny_seq))
    assert [i for i, _ in frames] == [1, 2, 3]
    assert frames[0][1].name == "000001.jpg"


def test_iter_frame_paths_window_and_bounds(tiny_seq):
    assert [i for i, _ in iter_frame_paths(tiny_seq, start=2, stop=10)] == [2, 3]
    with pytest.raises(ValueError):
        next(iter_frame_paths(tiny_seq, start=0))


def test_iter_frame_paths_missing_frame(tiny_seq):
    (tiny_seq / "img1" / "000002.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="missing frame 2"):
        list(iter_frame_paths(tiny_seq))
