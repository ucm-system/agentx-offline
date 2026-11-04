#!/usr/bin/env python3
"""Convert an AgentX / WEKA trace JSON into ClawPerf's kvcache.ai JSONL format.

ClawPerf's ``trace`` mode ingests one JSON object per request::

    {"hash_ids": [1,2,3], "input_length": 192, "block_size": 64, "user_id": 0}

AgentX traces store per-session turns (and nested subagent turns) with the same
``hash_ids`` / ``in`` fields, so the conversion is mostly a flatten + sort:

* every main request and every subagent inner request becomes one line;
* events are sorted by their timestamp ``t`` (seconds);
* ``user_id`` groups the session (main chain) and each subagent group, so
  ClawPerf's ``--trace-users`` can keep turns of one session ordered;
* the original AgentX JSON is preserved in ``agentx`` for traceability.

Usage:
  python scripts/convert_agentx_to_kvcache.py agentx/traces/agentx-syn-000000.json \
      -o /tmp/trace.jsonl
  python scripts/convert_agentx_to_kvcache.py --dataset 8 -o /tmp/agentx8.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _iter_requests(trace: dict[str, Any]):
    """Yield (t, hash_ids, in_tokens, user_key, kind) for all requests."""
    session = str(trace.get("id", "session"))
    for req in trace.get("requests", []):
        kind = req.get("type")
        if kind in ("n", "s"):
            yield (
                float(req.get("t", 0.0)),
                list(req.get("hash_ids") or []),
                int(req.get("in", 0)),
                session,
                "main",
            )
        elif kind == "subagent":
            agent_id = str(req.get("agent_id", "sub"))
            user = f"{session}:{agent_id}"
            for inner in req.get("requests", []):
                if inner.get("type") not in ("n", "s"):
                    continue
                yield (
                    float(inner.get("t", 0.0)),
                    list(inner.get("hash_ids") or []),
                    int(inner.get("in", 0)),
                    user,
                    "subagent",
                )


def convert_trace(trace: dict[str, Any]) -> list[dict[str, Any]]:
    block_size = int(trace.get("block_size") or 64)
    rows = []
    for t, hash_ids, in_tokens, user, kind in sorted(_iter_requests(trace), key=lambda x: x[0]):
        rows.append(
            {
                "hash_ids": hash_ids,
                "input_length": in_tokens,
                "block_size": block_size,
                "user_id": user,
                "agentx": {"trace_id": trace.get("id"), "t": t, "kind": kind},
            }
        )
    return rows


def load_dataset_rows(num: int, hf_home: str | None) -> list[dict[str, Any]]:
    if hf_home:
        import os

        os.environ.setdefault("HF_HOME", hf_home)
    from datasets import load_dataset

    ds = load_dataset("semianalysisai/cc-traces-weka-062126", split="train")
    return [ds[i] for i in range(min(num, len(ds)))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", help="AgentX trace JSON file")
    ap.add_argument("--dataset", type=int, default=0, help="convert first N dataset rows instead")
    ap.add_argument("--hf-home", default="")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)

    if args.dataset:
        traces = load_dataset_rows(args.dataset, args.hf_home or None)
    elif args.trace:
        traces = [json.loads(Path(args.trace).read_text(encoding="utf-8"))]
    else:
        print("provide a trace file or --dataset N", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for trace in traces:
            for row in convert_trace(trace):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += 1
    print(f"[convert] traces={len(traces)} requests={total} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
