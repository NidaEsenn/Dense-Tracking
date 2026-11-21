# ADR-001: New repository with a `src/` layout

**Status:** accepted · **Date:** 2026-11-20

## Context

The v3 course prototype ran as numbered scripts that modified `sys.path`, had no tests, and
loaded whole sequences into memory. Its evaluation also counted predictions on distractor
classes as false positives. The design report (v1.2) decided to start Odin's Eye in a new
repository (decision D10) rather than upgrade v3.

## Decision

- Installable package `odins_eye` under `src/`, configured in `pyproject.toml`.
- One sub-package per module (M1–M7) plus `io/`, `eval/` and `utils/`, added as each lands.
- Entry points in `scripts/`, numbered by pipeline stage.
- ruff + pytest enforced by pre-commit and GitHub Actions.
- Frames are streamed (`odins_eye.io.iter_frames`), never loaded all at once.

## Consequences

Modules can be tested in isolation and imported without path hacks. Only a few v3 parts are
copied in (Kalman filter, detector weights, MLX/Transformers loading), each with a note of origin.
