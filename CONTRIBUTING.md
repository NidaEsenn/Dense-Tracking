# Contributing

## Workflow

- `main` is protected; every change goes through a pull request with one approval and green CI.
- Branches: `feat/…`, `fix/…`, `exp/…`.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `chore:`.
- Keep PRs small (under ~400 changed lines) and include tests.
- Run `pre-commit install` once; ruff lint/format and pytest must pass.

## Conventions

- **Names:** `snake_case` for files, functions and variables; `PascalCase` for classes; scripts numbered by stage (`10_detect_track.py`, `20_crowd_state.py`, …).
- **Units and IDs:** pixels and seconds unless a homography gives meters; frames are 1-indexed like MOTChallenge; fps is read from `seqinfo.ini`, never assumed; track IDs are local to one video.
- **Schemas:** every JSON file carries `schema_version`; a schema change needs a version bump and a decision record in `docs/decisions/`.
- **Experiments:** one folder per run, `experiments/YYYYMMDD_<experiment-id>_<short-description>/`, holding the resolved config, git hash, `metrics.json` and logs (`odins_eye.utils.make_run_dir`). Reported numbers come only from such runs.
- **Data:** never commit data, weights or videos. Raw data is read-only; every derived file is produced by a script.
- **Privacy:** no face crops saved; released frames are face-blurred; no code that matches people across videos.
