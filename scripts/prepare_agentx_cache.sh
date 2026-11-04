#!/usr/bin/env bash
# Prepare the offline cache baked into the AgentX image (Linux/macOS).
# Copies the dataset Arrow cache + DeepSeek-V3 tokenizer cache into
# agentx_offline/cache so `docker build` can bake them in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT/data/hf}"
DST="${2:-$ROOT/agentx_offline/cache}"

DATASET="$SRC/datasets/semianalysisai___cc-traces-weka-062126"
TOKENIZER="$SRC/hub/models--deepseek-ai--DeepSeek-V3"

[ -d "$DATASET" ] || { echo "missing dataset cache: $DATASET" >&2; exit 1; }
[ -d "$TOKENIZER" ] || { echo "missing tokenizer cache: $TOKENIZER" >&2; exit 1; }

rm -rf "$DST"
mkdir -p "$DST/datasets" "$DST/hub"
cp -a "$DATASET" "$DST/datasets/"
cp -a "$TOKENIZER" "$DST/hub/"

du -sh "$DST"
echo "[cache] next: docker build -t agentx-offline:1.0.7 agentx_offline"
