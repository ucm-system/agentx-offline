#!/usr/bin/env python3
"""Offline AgentX runner: one target (an OpenAI-compatible LLM service) × N concurrency points.

This is the driver baked into the ``agentx-offline`` image.  It never touches
the network: the dataset Arrow cache and tokenizer are baked into the image
(``/opt/hf``), and ``HF_*_OFFLINE=1`` is set by the entrypoint.

Artifacts are laid out as::

    <out>/<target>/c<CONCURRENCY>/
        profile_export_aiperf.json      # aggregate metrics (official)
        profile_export.jsonl            # per-request records
        profile_export_aiperf_timeslices.*
        server_metrics_export.*
        profile_export_console.txt
        logs/aiperf.log
        run.log                         # our stdout/stderr capture

and ``<out>/report.html`` is regenerated after every point.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

AGENTX = os.environ.get("AGENTX_BIN", "agentx")
DEFAULT_DATASET = "semianalysis_cc_traces_weka_062126"
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"

# Supported context-window labels; values are tokens.  A small reserve is kept
# because AgentX filters on peak prompt+output and tokenizer reconstruction can
# differ by a few tokens from the trace's own count.
WINDOW_TOKENS = {
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "256k": 262144,
    "512k": 524288,
}


def window_tokens(label: str) -> int:
    key = label.lower()
    if key not in WINDOW_TOKENS:
        raise argparse.ArgumentTypeError(
            f"unknown window {label!r}; choose one of {', '.join(WINDOW_TOKENS)}"
        )
    return WINDOW_TOKENS[key]


def effective_max_context(args: argparse.Namespace) -> int:
    """Explicit --max-context wins, otherwise derive it from --window."""
    explicit = getattr(args, "max_context", 0) or 0
    if explicit:
        return explicit
    window = getattr(args, "window", "") or ""
    if window:
        return window_tokens(window) - 512
    return 0


def parse_concurrency(value: str) -> list[int]:
    points = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not points:
        raise argparse.ArgumentTypeError("at least one concurrency point is required")
    return points


def build_command(args: argparse.Namespace, concurrency: int, artifact_dir: Path) -> list[str]:
    cmd = [
        AGENTX,
        "profile",
        "--url",
        args.url,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--concurrency",
        str(concurrency),
        "--benchmark-duration",
        str(args.duration),
        "--num-dataset-entries",
        str(args.entries),
        "--stats-interval",
        str(args.stats_interval),
        "--output-artifact-dir",
        str(artifact_dir),
    ]
    if args.dataset_path:
        cmd += [
            "--input-file",
            args.dataset_path,
            "--custom-dataset-type",
            "weka_trace",
        ]
    else:
        cmd += ["--public-dataset", args.dataset]
    # One override is enough: local corpus identity + short duration are both
    # downgraded by the same flag.
    if args.dataset_path or args.duration < 900:
        cmd += ["--unsafe-override"]
    if args.trust_remote_code:
        cmd += ["--tokenizer-trust-remote-code"]
    if getattr(args, "max_context", 0):
        cmd += ["--max-context-length", str(args.max_context)]
    elif getattr(args, "window", ""):
        cmd += ["--max-context-length", str(effective_max_context(args))]
    if args.no_realtime:
        cmd += ["--no-realtime-metrics"]
    if args.extra:
        cmd += args.extra
    return cmd


def run_point(args: argparse.Namespace, target: str, concurrency: int) -> dict:
    artifact_dir = Path(args.out) / target / f"c{concurrency}"
    if artifact_dir.exists() and not args.force:
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, concurrency, artifact_dir)
    run_log = artifact_dir / "run.log"
    started = time.time()
    print(f"[agentx-offline] c={concurrency} -> {artifact_dir}", flush=True)
    print("[agentx-offline] " + " ".join(cmd), flush=True)
    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    export = artifact_dir / "profile_export_aiperf.json"
    ok = proc.returncode == 0 and export.exists() and export.stat().st_size > 0
    tail = ""
    if not ok:
        lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-12:])
    return {
        "target": target,
        "concurrency": concurrency,
        "exit_code": proc.returncode,
        "ok": ok,
        "artifact_dir": str(artifact_dir),
        "elapsed_s": round(elapsed, 1),
        "tail": tail,
    }


def point_stats(artifact_dir: Path) -> dict:
    """Key metrics for SLA checks, read from the official aggregate JSON."""
    summary = artifact_dir / "profile_export_aiperf.json"
    if not summary.exists() or summary.stat().st_size == 0:
        return {}
    data = json.loads(summary.read_text(encoding="utf-8"))

    def stat(key: str, which: str = "avg"):
        return (data.get(key) or {}).get(which)

    requests = int(stat("request_count") or 0)
    errors = len(data.get("error_summary") or [])
    return {
        "requests": requests,
        "success_pct": round(100.0 * (requests - errors) / requests, 2) if requests else 0.0,
        "out_tps": stat("output_token_throughput"),
        "rps": stat("request_throughput"),
        "ttft_p90": stat("time_to_first_token", "p90"),
        "ttft_p50": stat("time_to_first_token", "p50"),
        "itl_p50": stat("inter_token_latency", "p50"),
        "e2e_p90": stat("request_latency", "p90"),
    }


def sla_failures(stats: dict, args: argparse.Namespace) -> list[str]:
    """Return the list of violated SLA constraints (empty = pass)."""
    fails: list[str] = []
    if not stats:
        return ["no artifacts"]
    if stats["success_pct"] < args.sla_min_success:
        fails.append(f"success {stats['success_pct']:.1f}% < {args.sla_min_success}%")
    for key, limit, label in (
        ("ttft_p90", args.sla_ttft_p90, "TTFT p90"),
        ("itl_p50", args.sla_itl_p50, "ITL p50"),
        ("e2e_p90", args.sla_e2e_p90, "E2E p90"),
    ):
        if limit and (stats.get(key) or 0) > limit:
            fails.append(f"{label} {stats[key]:.0f} > {limit:.0f}")
    return fails


def select_best(rows: list[dict]) -> dict | None:
    """Highest-throughput row that satisfies the SLA; falls back to max tps."""
    passing = [r for r in rows if r.get("sla_ok") and (r.get("out_tps") or 0) > 0]
    if passing:
        return max(passing, key=lambda r: r["out_tps"])
    candidates = [r for r in rows if (r.get("out_tps") or 0) > 0]
    return max(candidates, key=lambda r: r["out_tps"]) if candidates else None


def auto_ladder(start: int, max_c: int) -> list[int]:
    values = []
    c = max(1, start)
    while c <= max_c:
        values.append(c)
        c *= 2
    if max_c not in values:
        values.append(max_c)
    return values


def should_stop(rows: list[dict], consecutive: int = 2) -> bool:
    """Stop the ladder after N consecutive SLA failures (or throughput drops)."""
    if len(rows) < 3:
        return False
    tail = rows[-consecutive:]
    if all(not r.get("sla_ok") for r in tail):
        return True
    peak = max((r.get("out_tps") or 0) for r in rows[:-consecutive])
    return peak > 0 and all((r.get("out_tps") or 0) < peak * 0.9 for r in tail)


def _fmt(value, digits: int = 0) -> str:
    if value is None:
        return "–"
    return f"{value:,.{digits}f}"


def _run_tune(args: argparse.Namespace) -> int:
    target = args.target or args.model
    points = auto_ladder(args.start, args.max_concurrency) if args.auto else args.concurrency
    rows: list[dict] = []
    print(f"[tune] target={target} window={args.window or args.max_context or 'full'} points={points}")
    for c in points:
        res = run_point(args, target, c)
        stats = point_stats(Path(res["artifact_dir"]))
        fails = sla_failures(stats, args) if stats else ["no artifacts"]
        rows.append(
            {
                "concurrency": c,
                "ok": res["ok"],
                "stats": stats,
                "sla_ok": not fails,
                "sla_failures": fails,
                "elapsed_s": res["elapsed_s"],
            }
        )
        print(
            f"  c={c:<4} tps={_fmt(stats.get('out_tps'), 1):>7} "
            f"ttft90={_fmt(stats.get('ttft_p90')):>7}ms itl50={_fmt(stats.get('itl_p50'), 1):>6}ms "
            f"sla={'PASS' if not fails else 'FAIL(' + '; '.join(fails) + ')'}"
        )
        if args.auto and should_stop(rows):
            print("[tune] stopping ladder: SLA failure or throughput dropped twice")
            break

    best = select_best(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "url": args.url,
        "model": args.model,
        "window": args.window,
        "max_context": effective_max_context(args),
        "sla": {
            "ttft_p90": args.sla_ttft_p90,
            "itl_p50": args.sla_itl_p50,
            "e2e_p90": args.sla_e2e_p90,
            "min_success": args.sla_min_success,
        },
        "rows": rows,
        "recommended": best,
    }
    (out / "tune.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# SLA 并发调优：{target}",
        "",
        f"- 上下文窗口：{args.window or args.max_context or '不限制'}",
        f"- SLA：TTFT p90 ≤ {args.sla_ttft_p90 or '∞'} ms，ITL p50 ≤ {args.sla_itl_p50 or '∞'} ms，"
        f"E2E p90 ≤ {args.sla_e2e_p90 or '∞'} ms，成功率 ≥ {args.sla_min_success}%",
        "",
        "| 并发 | 输出 tok/s | TTFT p90 (ms) | ITL p50 (ms) | E2E p90 (ms) | SLA | 失败原因 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["stats"]
        lines.append(
            f"| {r['concurrency']} | {_fmt(s.get('out_tps'), 1)} | {_fmt(s.get('ttft_p90'))} | "
            f"{_fmt(s.get('itl_p50'), 1)} | {_fmt(s.get('e2e_p90'))} | "
            f"{'PASS' if r['sla_ok'] else 'FAIL'} | {'; '.join(r['sla_failures'])} |"
        )
    if best:
        lines += [
            "",
            f"**推荐并发：c={best['concurrency']}，输出 {_fmt(best['stats'].get('out_tps'), 1)} tok/s**"
            f"（{'满足 SLA' if best['sla_ok'] else '无满足 SLA 的点，取吞吐最大'}）",
        ]
    (out / "tune.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report import build_report

    report_path = build_report(out, None)
    print(f"[tune] report: {report_path}")
    if best:
        print(
            f"[tune] recommended concurrency={best['concurrency']} "
            f"out_tps={_fmt(best['stats'].get('out_tps'), 1)} sla_ok={best['sla_ok']}"
        )
    return 0 if best else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentx-offline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run AgentX against one target")
    run.add_argument("--url", required=True, help="base URL or full /v1/chat/completions URL")
    run.add_argument("--model", required=True, help="model id the server actually sees")
    run.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    run.add_argument("--target", default="", help="label for this target (default: model)")
    run.add_argument("--dataset", default=DEFAULT_DATASET, help="public dataset alias")
    run.add_argument("--dataset-path", default="", help="local WEKA trace dir (marks run non-submission)")
    run.add_argument("--concurrency", type=parse_concurrency, default=[4], help="e.g. 4,8,16")
    run.add_argument("--duration", type=int, default=900, help="profiling seconds per point (official >=900)")
    run.add_argument("--entries", type=int, default=8, help="number of traces (393 needs a big machine)")
    run.add_argument(
        "--max-context",
        type=int,
        default=0,
        help="drop traces whose peak context exceeds this (match the served model's max_model_len)",
    )
    run.add_argument(
        "--window",
        default="",
        help="context window label (16k/32k/64k/128k/256k); sets --max-context automatically",
    )
    run.add_argument("--stats-interval", type=int, default=30)
    run.add_argument("--out", default="/results", help="results root (mounted volume)")
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--no-realtime", action="store_true")
    run.add_argument("--force", action="store_true", help="delete an existing point directory")
    run.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra agentx flags after --")

    rep = sub.add_parser("report", help="build report.html from an existing results root")
    rep.add_argument("--root", default="/results")
    rep.add_argument("--out", default="", help="output html path (default <root>/report.html)")

    tune = sub.add_parser("tune", help="sweep concurrency and pick max throughput under an SLA")
    tune.add_argument("--url", required=True)
    tune.add_argument("--model", required=True)
    tune.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    tune.add_argument("--target", default="")
    tune.add_argument("--dataset", default=DEFAULT_DATASET)
    tune.add_argument("--dataset-path", default="")
    tune.add_argument("--concurrency", type=parse_concurrency, default=[4, 8, 16, 32])
    tune.add_argument("--auto", action="store_true", help="start at --start and double until --max-concurrency")
    tune.add_argument("--start", type=int, default=1)
    tune.add_argument("--max-concurrency", type=int, default=128)
    tune.add_argument("--window", default="", help="context window label (16k/32k/64k/128k/256k)")
    tune.add_argument("--max-context", type=int, default=0)
    tune.add_argument("--duration", type=int, default=120)
    tune.add_argument("--entries", type=int, default=8)
    tune.add_argument("--stats-interval", type=int, default=30)
    tune.add_argument("--out", default="/results")
    tune.add_argument("--trust-remote-code", action="store_true")
    tune.add_argument("--no-realtime", action="store_true")
    tune.add_argument("--force", action="store_true")
    tune.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    tune.add_argument("--sla-ttft-p90", type=float, default=0, help="max TTFT p90 in ms (0 = ignore)")
    tune.add_argument("--sla-itl-p50", type=float, default=0, help="max ITL p50 in ms (0 = ignore)")
    tune.add_argument("--sla-e2e-p90", type=float, default=0, help="max E2E p90 in ms (0 = ignore)")
    tune.add_argument("--sla-min-success", type=float, default=99.0)

    args = p.parse_args(argv)

    if args.command == "report":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from report import build_report

        path = build_report(Path(args.root), Path(args.out) if args.out else None)
        print(f"[agentx-offline] report: {path}")
        return 0

    if args.command == "tune":
        return _run_tune(args)

    target = args.target or args.model
    results = [run_point(args, target, c) for c in args.concurrency]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report import build_report

    report_path = build_report(Path(args.out), None)

    summary = {
        "target": target,
        "url": args.url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dataset": args.dataset_path or args.dataset,
        "entries": args.entries,
        "duration_s": args.duration,
        "points": results,
        "report": str(report_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (Path(args.out) / target / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[agentx-offline] report: {report_path}")
    for r in results:
        print(f"  c={r['concurrency']:<3} ok={r['ok']} exit={r['exit_code']} {r['elapsed_s']}s")
        if not r["ok"] and r["tail"]:
            print("  --- tail of run.log ---")
            print("\n".join("  " + line for line in r["tail"].splitlines()))
    return 0 if all(r["ok"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
