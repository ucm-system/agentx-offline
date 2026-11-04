"""Tests for the offline AgentX HTML report (no network, no harness needed)."""

from __future__ import annotations

import json
from pathlib import Path

from agentx_offline.report import build_report, discover, render_html


def _make_point(root: Path, target: str, conc: int, ttft: float, tps: float) -> None:
    d = root / target / f"c{conc}"
    (d / "logs").mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "request_count": {"avg": 10, "unit": "requests"},
                "request_throughput": {"avg": 0.5, "unit": "requests/sec"},
                "time_to_first_token": {"p50": ttft, "p90": ttft * 1.4, "p99": ttft * 1.8},
                "inter_token_latency": {"p50": 8.0, "p90": 9.0},
                "request_latency": {"p50": 1200, "p90": 2400, "p99": 3000},
                "output_token_throughput": {"avg": tps, "unit": "tokens/sec"},
                "input_token_throughput": {"avg": 1000.0},
                "input_sequence_length": {"p50": 54000.0},
                "output_sequence_length": {"p50": 120.0},
                "error_summary": [],
            }
        ),
        encoding="utf-8",
    )
    (d / "profile_export_console.txt").write_text(
        "│               Theoretical Prefix Cache Hit (%) │     95.53 │\n", encoding="utf-8"
    )
    (d / "logs" / "aiperf.log").write_text(
        "records_manager - INFO -   srv    prefix_cache_hit=93.4% unique_in_srv=1\n", encoding="utf-8"
    )
    (d / "profile_export_aiperf_timeslices.csv").write_text(
        "Timeslice,Start_NS,End_NS,Metric,Unit,Stat,Value\n"
        "0,0,1,Time to First Token,ms,p50,160\n"
        "1,0,1,Time to First Token,ms,p50,170\n"
        "0,0,1,Output Token Throughput,tokens/sec,avg,100\n"
        "1,0,1,Output Token Throughput,tokens/sec,avg,110\n",
        encoding="utf-8",
    )
    records = [
        {
            "metadata": {"benchmark_phase": "profiling"},
            "metrics": {
                "time_to_first_token": {"value": ttft},
                "request_latency": {"value": 1200.0},
            },
        },
        {
            "metadata": {"benchmark_phase": "profiling"},
            "metrics": {
                "time_to_first_token": {"value": ttft * 2},
                "request_latency": {"value": 2400.0},
            },
        },
        {
            "metadata": {"benchmark_phase": "warmup"},
            "metrics": {
                "time_to_first_token": {"value": 9999.0},
                "request_latency": {"value": 9999.0},
            },
        },
    ]
    (d / "profile_export.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def test_discover_and_render(tmp_path: Path) -> None:
    _make_point(tmp_path, "qwen3-8b", 4, 160.0, 103.0)
    _make_point(tmp_path, "llama-3-8b", 4, 260.0, 80.0)
    points, targets = discover(tmp_path)
    assert targets == ["llama-3-8b", "qwen3-8b"]
    assert len(points) == 2
    p = next(p for p in points if p["target"] == "qwen3-8b")
    assert p["cache_theory_pct"] == 95.53
    assert p["cache_server_pct"] == 93.4
    assert p["timeline"]["ttft_p50"] == [160.0, 170.0]
    # CDF ignores warmup records and exposes profiling-phase percentiles
    assert p["cdf"]["ttft"][-1][1] == 100.0
    assert max(x[0] for x in p["cdf"]["ttft"]) == 320.0
    assert p["interactivity"] == 125.0

    html = render_html(points, targets, "/* echarts stub */")
    assert "AgentX 推理性能对比报告" in html
    assert "qwen3-8b" in html and "llama-3-8b" in html
    assert 'id="c-ttft50"' in html
    assert 'id="c-cdf-ttft"' in html
    assert 'id="x-metric"' in html and 'id="y-metric"' in html
    assert 'id="c-explorer"' in html
    assert '"defaultX": "out_tps"' in html
    assert '"defaultY": "interactivity"' in html
    assert "echarts" in html
    assert "95.5%" in html


def test_build_report_writes_files(tmp_path: Path) -> None:
    _make_point(tmp_path, "target-a", 4, 100.0, 120.0)
    out = build_report(tmp_path)
    assert out.exists() and out.stat().st_size > 1000
    data = json.loads((tmp_path / "report_data.json").read_text(encoding="utf-8"))
    assert data["targets"] == ["target-a"]
    assert data["points"][0]["target"] == "target-a"


def test_discover_ignores_incomplete(tmp_path: Path) -> None:
    _make_point(tmp_path, "ok", 4, 100.0, 100.0)
    (tmp_path / "bad" / "c4").mkdir(parents=True)
    (tmp_path / "bad" / "c4" / "profile_export_aiperf.json").write_text("", encoding="utf-8")
    points, targets = discover(tmp_path)
    assert targets == ["ok"]
    assert len(points) == 1
