# ClawPerf Benchmark Report

| Field | Value |
|-------|-------|
| Model | `` |
| Endpoint | `` |
| Backend | vllm |
| Mode | `trace` |
| Requests | 77 |
| Total Tokens | 742,774 |
| Unique Blocks | 186 |
| Policy | lru |
| Setup Time | 0.00s |
| Bench Time | 0.00s |

## Verdict: ✅ GOOD


## Key Findings

- KV cache hit rate: 98.36%  (ceiling: 98.36%)
- Ideal prefill speedup: 60.85x  (= 1 / (1 - 98.36%))

## Summary

| Metric | Value |
|--------|-------|
| Requests | 77 |
| Total Input Tokens | 742,774 |
| Unique Blocks | 186 |
| Hit Rate | 98.36% |
| Ceiling | 98.36% |
| Speedup | 60.85x |
| Hit Tokens | 712,448 |
| Miss Tokens | 11,904 |

## Methodology

<details>
<summary>Click to expand</summary>

- **TTFT** (Time to First Token): wall time from request send to first content chunk on the wire.
- **TPOT** (Time Per Output Token): decode time / output tokens, excluding prefill.
- **ITL** (Inter-Token Latency): gap between consecutive output chunks.
- **Prefix cache hit rate**: token-level, read from the backend's Prometheus counters (start/end delta). Not request-level.
- **Verdict thresholds**: TTFT GOOD ≤3s / OK ≤10s; throughput GOOD ≥30 tok/s / OK ≥15 tok/s.
- **Compaction**: when context exceeds ``max_context_tokens``, history is cleared and the user prefix is incremented.
- **Decode throughput** isolates generation speed from prefill (excludes TTFT).
- **Wall-clock per-user throughput** uses real start/end timestamps, not summed per-request latencies.

</details>
