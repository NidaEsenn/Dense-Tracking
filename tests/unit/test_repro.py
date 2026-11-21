from __future__ import annotations

import datetime as dt
import json
import random

import numpy as np
import pytest

from odins_eye.utils import git_hash, make_run_dir, seed_everything


def test_seed_everything_is_deterministic():
    seed_everything(123)
    a = (random.random(), np.random.rand())  # noqa: NPY002
    seed_everything(123)
    b = (random.random(), np.random.rand())  # noqa: NPY002
    assert a == b


def test_git_hash_outside_repo(tmp_path):
    assert git_hash(tmp_path) == "unknown"


def test_make_run_dir_layout(tmp_path):
    run = make_run_dir(
        tmp_path,
        "e0",
        "trackeval-rescore",
        config={"tracker": "bytetrack"},
        date=dt.date(2026, 11, 20),
    )
    assert run.name == "20261120_e0_trackeval-rescore"
    meta = json.loads((run / "run.json").read_text())
    assert meta["config"] == {"tracker": "bytetrack"}
    assert "git_hash" in meta


def test_make_run_dir_refuses_overwrite(tmp_path):
    make_run_dir(tmp_path, "e1", "botsort", date=dt.date(2026, 11, 20))
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "e1", "botsort", date=dt.date(2026, 11, 20))


@pytest.mark.parametrize("bad", ["E1", "my run", "x_y", ""])
def test_make_run_dir_rejects_bad_names(tmp_path, bad):
    with pytest.raises(ValueError):
        make_run_dir(tmp_path, bad, "ok")
