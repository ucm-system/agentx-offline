#!/usr/bin/env bash
# Offline AgentX test image entrypoint.
#
#   docker run --rm -v "$PWD/results:/results" agentx-offline run \
#     --url http://llm.internal:8000 --model Qwen3-8B \
#     --concurrency 4,8,16 --duration 900
#
# Everything is already on disk: dataset Arrow cache and tokenizer live under
# $HF_HOME (/opt/hf), so the container never needs network access.
set -euo pipefail

export HF_HOME="${HF_HOME:-/opt/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1

# AIPerf's Windows-only ZMQ fallback needs a free port window; harmless on Linux.
export AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT="${AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT:-20000}"
export AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE="${AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE:-6000}"

usage() {
  cat <<'EOF'
agentx-offline — run the official AgentX harness against any OpenAI-compatible LLM service, fully offline.

Commands:
  run      Run AgentX for one target over one or more concurrency points, then render report.html
           required: --url --model
           common:   --tokenizer (default deepseek-ai/DeepSeek-V3) --concurrency 4,8,16
                     --duration 900 --entries 8 --out /results --target <label>
  report   Re-render report.html from an existing results root:
           report --root /results
  shell    Drop into bash inside the image (debugging)
  smoke    Self-check: dataset + tokenizer load offline, prints versions

Examples:
  docker run --rm -v "$PWD/results:/results" agentx-offline \
    run --url http://llm:8000 --model Qwen3-8B --concurrency 4,8,16 --duration 900

  docker run --rm -v "$PWD/results:/results" agentx-offline \
    report --root /results
EOF
}

case "${1:-}" in
  run|report)
    exec python3 /opt/agentx/runner.py "$@"
    ;;
  report-standalone)
    shift
    exec python3 /opt/agentx/report.py "$@"
    ;;
  shell)
    exec bash
    ;;
  smoke)
    exec python3 - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset
from aiperf.common.tokenizer import Tokenizer

repo = "semianalysisai/cc-traces-weka-062126"
print("HF_HOME:", os.environ["HF_HOME"])
ds = load_dataset(repo, split="train")
print("dataset rows:", len(ds))
tok = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
print("tokenizer ok, tokens(hello world):", len(tok.encode("hello world")))
print("offline smoke OK")
PY
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    exec "$@"
    ;;
esac
