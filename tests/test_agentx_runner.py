"""Tests for the offline AgentX runner command construction (no harness needed)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentx_offline.runner import (
    auto_ladder,
    build_command,
    effective_max_context,
    parse_concurrency,
    select_best,
    should_stop,
    sla_failures,
    window_tokens,
)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        url="http://llm:8000",
        model="Qwen3-8B",
        tokenizer="deepseek-ai/DeepSeek-V3",
        dataset="semianalysis_cc_traces_weka_062126",
        dataset_path="",
        duration=900,
        entries=8,
        stats_interval=30,
        max_context=0,
        window="",
        sla_ttft_p90=0,
        sla_itl_p50=0,
        sla_e2e_p90=0,
        sla_min_success=99.0,
        trust_remote_code=False,
        no_realtime=False,
        extra=[],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_public_dataset_official_duration() -> None:
    cmd = build_command(_args(), 8, Path("/r/c8"))
    assert "--public-dataset" in cmd
    assert cmd[cmd.index("--public-dataset") + 1] == "semianalysis_cc_traces_weka_062126"
    assert "--unsafe-override" not in cmd
    assert cmd[cmd.index("--concurrency") + 1] == "8"
    assert cmd[cmd.index("--benchmark-duration") + 1] == "900"


def test_short_duration_adds_single_override() -> None:
    cmd = build_command(_args(duration=120), 4, Path("/r/c4"))
    assert cmd.count("--unsafe-override") == 1


def test_local_trace_mode_adds_single_override() -> None:
    cmd = build_command(_args(dataset_path="/traces", duration=120), 2, Path("/r/c2"))
    assert cmd.count("--unsafe-override") == 1
    assert cmd[cmd.index("--input-file") + 1] == "/traces"
    assert cmd[cmd.index("--custom-dataset-type") + 1] == "weka_trace"
    assert "--public-dataset" not in cmd


def test_extra_passthrough_and_flags() -> None:
    cmd = build_command(
        _args(trust_remote_code=True, no_realtime=True, extra=["--chips", "8"]), 4, Path("/r/c4")
    )
    assert "--tokenizer-trust-remote-code" in cmd
    assert "--no-realtime-metrics" in cmd
    assert cmd[-2:] == ["--chips", "8"]


def test_parse_concurrency() -> None:
    assert parse_concurrency("1,4, 16") == [1, 4, 16]


def test_max_context_passthrough() -> None:
    cmd = build_command(_args(max_context=130000), 4, Path("/r/c4"))
    assert cmd[cmd.index("--max-context-length") + 1] == "130000"
    cmd_default = build_command(_args(), 4, Path("/r/c4"))
    assert "--max-context-length" not in cmd_default


def test_window_sets_max_context() -> None:
    assert window_tokens("64k") == 65536
    assert window_tokens("256k") == 262144
    args = _args(window="64k")
    assert effective_max_context(args) == 65536 - 512
    cmd = build_command(args, 8, Path("/r/c8"))
    assert cmd[cmd.index("--max-context-length") + 1] == str(65536 - 512)


def test_explicit_max_context_beats_window() -> None:
    assert effective_max_context(_args(window="64k", max_context=60000)) == 60000


def test_auto_ladder() -> None:
    assert auto_ladder(1, 128) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert auto_ladder(2, 100) == [2, 4, 8, 16, 32, 64, 100]


def test_select_best_prefers_sla() -> None:
    rows = [
        {"concurrency": 4, "out_tps": 100.0, "sla_ok": True},
        {"concurrency": 8, "out_tps": 150.0, "sla_ok": False},
    ]
    assert select_best(rows)["concurrency"] == 4
    assert select_best([{"concurrency": 4, "out_tps": 100.0, "sla_ok": False}])["concurrency"] == 4


def test_should_stop_after_two_sla_failures() -> None:
    rows = [
        {"sla_ok": True, "out_tps": 10.0},
        {"sla_ok": False, "out_tps": 9.0},
        {"sla_ok": False, "out_tps": 8.0},
    ]
    assert should_stop(rows) is True
    rows[2]["sla_ok"] = True
    assert should_stop(rows) is False


def test_sla_failures() -> None:
    args = _args(sla_min_success=99, sla_ttft_p90=1000, sla_itl_p50=50)
    stats = {"success_pct": 100.0, "ttft_p90": 800.0, "itl_p50": 60.0, "e2e_p90": 99999.0}
    fails = sla_failures(stats, args)
    assert len(fails) == 1 and "ITL" in fails[0]
