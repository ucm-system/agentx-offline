"""Populate $HF_HOME with the AgentX corpus + tokenizer (online prep step).

Used when building the offline image on a build host: run it once with network,
then ``docker build`` bakes the resulting ``datasets/`` and ``hub/`` into the
image.  Everything afterwards runs with ``HF_*_OFFLINE=1``.

    HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/agentx/agentx_offline/cache \
      python prefetch_cache.py
"""

from __future__ import annotations

import os
import sys
import time


def main() -> int:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("HF_HOME must point at the cache directory", file=sys.stderr)
        return 2
    print(f"[prefetch] HF_HOME={hf_home}", flush=True)
    print(f"[prefetch] HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}", flush=True)

    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset("semianalysisai/cc-traces-weka-062126", split="train")
    print(f"[prefetch] dataset rows={len(ds)} in {time.time() - t0:.1f}s", flush=True)
    print(f"[prefetch] arrow={ds.cache_files}", flush=True)

    from aiperf.common.tokenizer import Tokenizer

    t1 = time.time()
    tok = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", resolve_alias=False)
    print(f"[prefetch] tokenizer ok in {time.time() - t1:.1f}s ({len(tok.encode('hello world'))} tok)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
