# Qwen3.5-0.8B 并发扫描（1 → 128，真实服务）

目标：找出吞吐收益最大的并发点。服务：`vllm-ascend:v0.23.0` + `/mnt/model/Qwen3.5-0.8B`，
独占 **davinci7 一张 910B3**，`--max-num-seqs 128`、`--max-model-len 16384`、`gpu-memory-utilization 0.85`、
bf16 + MTP(1) + prefix caching；AgentX 用 `agentx/traces_13k`（8 条合成 trace，峰值 ≤13k），
每点 120s、`--max-context 16000`。

## 结果

| 并发 | 请求 | TTFT p50 (ms) | ITL p50 (ms) | 输出 tok/s | 每翻倍增益 | 每会话 tok/s | E2E p90 (ms) | 缓存（理论/服务端） |
|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 893 | 10.1 | 43.8 | – | 43.8 | 15,730 | 100% / 99.8% |
| 2 | 20 | 833 | 10.3 | 64.9 | +48.3% | 32.5 | 12,331 | 100% / 99.8% |
| 4 | 39 | 1,111 | 12.3 | 99.0 | +52.4% | 24.7 | 12,267 | 100% / 99.9% |
| 8 | 62 | 1,241 | 24.2 | 152.9 | +54.4% | 19.1 | 21,475 | 100% / 99.9% |
| **16** | 88 | 1,536 | 47.9 | **196.1** | +28.3% | 12.3 | 30,916 | 99.0% / 99.6% |
| 32 | 111 | 4,193 | 93.2 | 190.9 | −2.7% | 6.0 | 46,861 | 99.1% / 99.6% |
| 64 | 109 | 9,538 | 116.5 | 132.8 | −30.5% | 2.1 | 57,365 | 99.6% / 99.7% |
| 128 | – | – | – | – | – | – | – | 失败（见下） |

## 结论

- **吞吐峰值在 c=16：196.1 tok/s**。c=8→16 还能拿到 +28%/翻倍，c=16→32 已经是 −2.7%，
  再往上（64）直接掉到 132.8 tok/s；
- **收益最大的区间是 c=2…8**（每翻倍 +48%…+54%）；但从“绝对吞吐最大”看选 **c=16**；
- **延迟敏感场景选 c=8**：ITL 24.2ms、TTFT p50 1.24s，仍能保持 153 tok/s；
- c=32 之后 TTFT/ITL/E2E 全面劣化（ITL 93ms、E2E p90 47s），说明单卡 0.8B 的调度/带宽已经饱和；
- **c=128 失败**：服务端 `EngineDeadError`（KV/显存打满，大量 500），AIPerf `ProfileAborted`。
  这不是脚本问题，是 128 个 ~11k token 会话超出单卡 64GB 的 KV 预算；同时 AgentX 的 128 lane
  还有 435 条强制 snapshot primer，warmup 本身就要 ~8 分钟（已用 `--warmup-requests-per-lane 1` 压过）。

## 复现

```bash
# 服务（独占 chip 7）
vllm serve /mnt/model/Qwen3.5-0.8B --host 127.0.0.1 --port 8080 \
  --max-num-seqs 128 --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 --trust-remote-code \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' ...

# 扫描（镜像内）
docker run --rm --net host -v /mnt/model:/mnt/model:ro -v /data/model:/data/model:ro \
  -v /data/agentx/traces_13k:/traces:ro -v /data/agentx/results_sweep:/results \
  ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64 run \
    --url http://127.0.0.1:8080 --model qwen3.5 \
    --tokenizer /data/model/Qwen3.5-0.8B --target qwen35-sweep \
    --dataset-path /traces --max-context 16000 \
    --concurrency 1,2,4,8,16,32,64 --duration 120 --entries 8

# 拐点分析
python scripts/analyze_agentx_sweep.py results/real_qwen35_sweep/report_data.json \
  --target qwen35-sweep
```

产物：`report.html`（含 X/Y 曲线浏览器）、`report_data.json`、`preview.png`；原始 artifacts 在服务器
`/data/agentx/results_sweep/qwen35-sweep/c<N>/`。
