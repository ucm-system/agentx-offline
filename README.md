# agentx-offline

Run the official [AgentX](https://inferencex.semianalysis.com/agentx) agentic benchmark
against any OpenAI-compatible LLM service, **fully offline**, and turn the official
artifacts into a single-file HTML comparison report (with an InferenceX-style X/Y
explorer chart, per-request CDFs and SLA-based concurrency auto-tuning).

The image bakes the official `agentx-harness==1.0.7`, the pinned corpus
`semianalysisai/cc-traces-weka-062126` (393 traces, 1.77 GB Arrow) and the
DeepSeek-V3 tokenizer, so the target machine needs no internet: only the service
under test has to be reachable.

## What you get

- `agentx-offline run` — one target × N concurrency points, official
  `profile_export_aiperf.json` / `profile_export.jsonl` / timeslices /
  `server_metrics_export.json` per point;
- `agentx-offline report` — self-contained `report.html` (ECharts inlined, no CDN):
  KPI cards, summary table, 6 comparison charts, TTFT/E2E CDFs, per-second timeline,
  and an **X/Y switchable explorer** (throughput, interactivity, TTFT p50/p90/p99,
  ITL, E2E, RPS, cache hit, ISL/OSL, concurrency; log axis; scatter/line);
- `agentx-offline tune` — sweep concurrency, apply SLA thresholds
  (`--sla-ttft-p90`, `--sla-itl-p50`, `--sla-e2e-p90`, `--sla-min-success`), pick the
  max-throughput point that still meets the SLA, and write `tune.json` / `tune.md`
  plus a fresh report;
- `--window 64k|128k|256k|512k` — context-window-aware trace filtering (AgentX
  `--max-context-length`) for models that cannot take the full 142k-median corpus;
- `scripts/convert_agentx_to_kvcache.py` — export an AgentX trace to ClawPerf's
  kvcache.ai JSONL for KV-cache hit-rate / budget analysis (see
  `results/clawperf_agentx/`).

## Quick start

```bash
# offline package: load the image on the isolated machine
docker pull ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64   # or docker load -i ...

docker run --rm agentx-offline:1.0.7-arm64 smoke      # corpus + tokenizer, no network

docker run --rm --net host -v /srv/results:/results \
  agentx-offline:1.0.7-arm64 run \
    --url http://127.0.0.1:8080 --model qwen3.5 \
    --tokenizer /mnt/model/Qwen3.5-0.8B \
    --target qwen35 --concurrency 4,8,16 --duration 900

docker run --rm -v /srv/results:/results agentx-offline:1.0.7-arm64 report --root /results
```

- `--url` OpenAI-compatible base URL (or full `/v1/chat/completions`);
- `--model` the model id the server sees;
- `--tokenizer` HF id, **local model/tokenizer directory**, or `builtin`;
- `--duration` ≤900 s is allowed but flagged (`--unsafe-override`), official
  submissions require ≥900 s;
- `--entries` number of traces (393 needs a big host; 8–64 is fine for smoke tests).

## Build

```bash
# prepare the 1.8 GB corpus/tokenizer cache once (needs network)
bash scripts/prepare_agentx_cache.sh          # or pwsh scripts/prepare_agentx_cache.ps1

# native build
docker build -t agentx-offline:1.0.7 agentx_offline

# cross-build arm64 (needs an arm64 Python base image, e.g. pulled on the target host)
docker buildx build --platform linux/arm64 \
  --build-arg BASE_IMAGE=python:3.11-slim \
  -t agentx-offline:1.0.7-arm64 agentx_offline
```

The Dockerfile is multi-stage: the builder installs a Rust toolchain (aiperf's
`crick` dependency has no linux-aarch64 wheel) and produces a wheelhouse; the final
stage only installs wheels, drops privileges to uid 10001, and bakes `/opt/hf`.

A GitHub Actions workflow (`.github/workflows/agentx-image.yml`) builds amd64 on
`ubuntu-24.04` and arm64 on `ubuntu-24.04-arm`, optionally pre-fetches the corpus
(`with_cache=true`) and pushes to GHCR with a multi-arch manifest.

## Context windows

AgentX filters traces whose **peak prompt+output** exceeds the served model's
window. Eligibility of the 393 official traces:

| window | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|
| usable traces | 0 | 9 | 85 | 221 | 323 | 393 |

```bash
agentx-offline run ... --window 128k --entries 16
```

Small windows have very few traces — concurrency curves get noisy above a handful of
sessions; use local traces sized to the window for dense load sweeps.

## SLA auto-tuning

```bash
agentx-offline tune \
  --url http://127.0.0.1:8080 --model qwen3.5 \
  --tokenizer /mnt/model/Qwen3.5-0.8B \
  --target qwen35-128k --window 128k \
  --concurrency 2,4,8,16 --duration 120 --entries 8 \
  --sla-ttft-p90 30000 --sla-itl-p50 50 --sla-min-success 99
# or adaptive: --auto --start 1 --max-concurrency 128 (stops after 2 failures)
```

The tuner reads the official aggregate JSON per point, marks PASS/FAIL, recommends the
highest-throughput PASS point, and writes `tune.json`, `tune.md`, `report.html`.

## Real measurements

On one Ascend 910B3 (Qwen3.5-0.8B, `vllm-ascend:v0.23.0`), 120 s per point:

| workload | peak | at | note |
|---|---|---|---|
| 13k synthetic traces | 196 tok/s | c=16 | short-context ceiling; c=32 drops |
| 128K official subset (85 traces) | 73.7 tok/s | c=4 | TTFT p90 8.8 s, ITL 21.5 ms |
| 256K official subset (221 traces) | 56.0 tok/s | c=8 | TTFT p90 17.6 s |
| 64K official subset (9 traces) | ~56–64 tok/s | noisy | 64K prefill dominates |

Reports and tune logs: `results/real_qwen35*/`. These are single-card, shared-cluster
snapshots — use them to validate the pipeline and relative trends, not as a leaderboard.

## ClawPerf interop

`scripts/convert_agentx_to_kvcache.py` flattens an AgentX/WEKA session (including
subagent turns) into ClawPerf's trace format; ClawPerf's `trace` mode then reproduces
AgentX's theoretical prefix-cache hit **exactly** (98.36% on the example trace), but
real request replay needs `messages` (AgentX traces carry only `hash_ids`), and
latency/throughput are not comparable because ClawPerf ignores AgentX's inter-turn
timing, subagent DAG, warmup and trajectory slicing. Details: `results/clawperf_agentx/`.

## Layout

```
agentx_offline/        image sources (Dockerfile, runner, report, entrypoint, patch)
scripts/               cache prep, kvcache converter, sweep analysis
docs/                  AgentX methodology, real-run notes, image guide
results/               selected real reports, tune logs, ClawPerf comparison
tests/                 report/runner/converter/sweep unit tests
.github/workflows/     multi-arch image build
```

## Caveats

- Mock/score interpretation: AgentX numbers are only meaningful against real models;
  the image itself does not care what is behind `--url`.
- The official corpus is long-context (median ISL 142k); small-memory hosts should use
  `--entries` and `--window`.
- AgentX is a closed-loop agent replay: `--concurrency` is concurrent agent sessions,
  not HTTP batches, and request completion counts differ per point.

Apache-2.0.
