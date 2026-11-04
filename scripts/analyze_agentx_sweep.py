#!/usr/bin/env python3
"""Find the throughput saturation point from an AgentX report_data.json.

Usage:
  python scripts/analyze_agentx_sweep.py results/real_qwen35_sweep/report_data.json \
      --target qwen35-sweep

For each concurrency point it prints output tok/s, per-doubling gain and
per-session efficiency, then recommends the smallest concurrency whose
marginal gain per doubling falls below --threshold (default 10%).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_points(path: Path, target: str | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = [p for p in data["points"] if target is None or p["target"] == target]
    points.sort(key=lambda p: p["concurrency"])
    return points


def analyze(points: list[dict], threshold: float = 0.10) -> dict:
    rows = []
    prev = None
    for p in points:
        tps = p.get("out_tps")
        conc = p["concurrency"]
        gain = None
        if prev and prev.get("out_tps") and tps:
            ratio = math.log2(conc / prev["concurrency"])
            if ratio > 0:
                gain = (tps / prev["out_tps"]) ** (1.0 / ratio) - 1.0
        rows.append(
            {
                "concurrency": conc,
                "requests": p.get("requests"),
                "ttft_p50": (p.get("ttft") or {}).get("p50"),
                "itl_p50": (p.get("itl") or {}).get("p50"),
                "out_tps": tps,
                "gain_per_doubling": gain,
                "tps_per_session": (tps / conc) if tps else None,
            }
        )
        prev = p

    knee = None
    for row in rows:
        if row["gain_per_doubling"] is not None and row["gain_per_doubling"] < threshold:
            knee = row["concurrency"]
            break
    peak = max((r for r in rows if r["out_tps"]), key=lambda r: r["out_tps"], default=None)
    return {"rows": rows, "knee": knee, "peak": peak, "threshold": threshold}


def render_markdown(result: dict) -> str:
    lines = [
        "| 并发 | 请求 | TTFT p50 (ms) | ITL p50 (ms) | 输出 tok/s | 每翻倍增益 | 每会话 tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        gain = "–" if r["gain_per_doubling"] is None else f"{r['gain_per_doubling'] * 100:.1f}%"
        eff = "–" if r["tps_per_session"] is None else f"{r['tps_per_session']:.1f}"
        lines.append(
            f"| {r['concurrency']} | {r['requests']} | {r['ttft_p50']:.0f} | "
            f"{r['itl_p50']:.1f} | {r['out_tps']:.1f} | {gain} | {eff} |"
        )
    knee = result["knee"]
    peak = result["peak"]
    if peak:
        lines.append("")
        lines.append(f"- 峰值吞吐：c={peak['concurrency']} 时 {peak['out_tps']:.1f} tok/s")
    if knee:
        lines.append(
            f"- 推荐拐点：c={knee}（此后每翻倍吞吐增益 < {result['threshold'] * 100:.0f}%）"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="report_data.json")
    ap.add_argument("--target", default=None)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    points = load_points(args.data, args.target)
    if not points:
        print("no points", file=sys.stderr)
        return 1
    result = analyze(points, args.threshold)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
