"""Tests for the concurrency-sweep saturation analysis."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_agentx_sweep import analyze, render_markdown


def _point(c: int, tps: float) -> dict:
    return {
        "concurrency": c,
        "requests": 10,
        "out_tps": tps,
        "ttft": {"p50": 100.0},
        "itl": {"p50": 10.0},
    }


def test_knee_detected_when_plateau() -> None:
    # Each doubling gains >10% until c=8, then 300 -> 310 is only ~3% per doubling.
    pts = [_point(1, 50), _point(2, 95), _point(4, 170), _point(8, 300), _point(16, 310)]
    result = analyze(pts, threshold=0.10)
    assert result["knee"] == 16
    assert result["peak"]["concurrency"] == 16
    md = render_markdown(result)
    assert "| 8 |" in md
    assert "峰值吞吐" in md


def test_no_knee_when_every_step_grows() -> None:
    pts = [_point(1, 50), _point(2, 100), _point(4, 200), _point(8, 400)]
    result = analyze(pts, threshold=0.10)
    assert result["knee"] is None
    assert result["peak"]["out_tps"] == 400


def test_gain_normalised_by_doublings() -> None:
    # c=1 -> 4 is two doublings; 100 -> 121 is exactly 10% per doubling.
    pts = [_point(1, 100), _point(4, 121)]
    result = analyze(pts, threshold=0.10)
    gain = result["rows"][1]["gain_per_doubling"]
    assert abs(gain - 0.10) < 1e-6
