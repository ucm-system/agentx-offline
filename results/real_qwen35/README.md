# 真实模型实测：Qwen3.5-0.8B on Ascend 910B3（chip 7）

用推送到 CCR 的 `agentx-offline:1.0.7-arm64` 镜像，对 **真实 vLLM-Ascend 服务** 跑 AgentX；
两个实例都只占 **davinci7 一张卡**，顺序启动（共享一张卡时同时起两个会 OOM，见下）。

## 服务配置（`quay.io/ascend/vllm-ascend:v0.23.0`）

| 实例 | 端口 | 关键参数 |
|---|---|---|
| `qwen35-mtp` | 8080 | MTP 投机解码 `num_speculative_tokens=1`、`FULL_DECODE_ONLY` cudagraph `[2,16]`、prefix caching 开、`gpu-memory-utilization=0.40`、`max-num-seqs=32`、`max-model-len=16384` |
| `qwen35-base` | 8095 | 无 MTP、无 cudagraph、`--no-enable-prefix-caching`、`gpu-memory-utilization=0.30`、其余同上 |

两个实例都必须用 `--dtype bfloat16`：`float16` 下 Qwen3.5 的 GDN 算子
（`aclnnChunkGatedDeltaRuleFwdH`）报 dtype 不匹配，服务起不来。

## AgentX 运行方式

官方 393 条语料的峰值上下文都 >32k（最小的一条也超过 32k），`max-model-len=16384` 的服务
装不下，因此用本地合成 WEKA trace（`agentx/traces_13k`，8 条 session、750 请求、88 子 agent，
峰值 ≤13k），`--max-context 16000` 过滤：

```bash
docker run --rm --net host \
  -v /mnt/model:/mnt/model:ro -v /data/model:/data/model:ro \
  -v /data/agentx/traces_13k:/traces:ro -v /data/agentx/real_results:/results \
  ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64 run \
    --url http://127.0.0.1:8080 --model qwen3.5 \
    --tokenizer /data/model/Qwen3.5-0.8B \
    --target qwen35-mtp --dataset-path /traces --max-context 16000 \
    --concurrency 4,8 --duration 120 --entries 8
```

`--tokenizer` 直接指向本地模型目录（镜像已 patch aiperf 1.0.7 的本地目录 bug；上游只支持
HF cache id）。本地 trace 模式会打 `submission_valid=false`，其余 artifacts/报告流程与官方一致。

## 结果（真实运行，120s/点）

| target | 并发 | 请求 | TTFT p50/p90 (ms) | ITL p50 (ms) | 输出 tok/s | E2E p90 (ms) | 缓存命中（理论/服务端） |
|---|---|---|---|---|---|---|---|
| qwen35-base | 4 | 32 | 3,834 / 5,638 | 22.4 | 82.9 | 21,894 | 100% / 99.9% |
| qwen35-base | 8 | 59 | 1,214 / 4,957 | 22.1 | 152.3 | 29,190 | 100% / 99.9% |
| qwen35-mtp | 4 | 29 | 1,709 / 8,458 | 15.3 | 81.6 | 25,108 | 100% / 99.9% |
| qwen35-mtp | 8 | 30 | 4,775 / 12,866 | 63.6 | 80.3 | 46,988 | 100% / 99.8% |

## 怎么解读（重要）

- **MTP 在低并发有效**：c=4 时 ITL 从 22.4ms 降到 15.3ms（约 1.46×），TTFT p50 也低一些；
- **MTP 在高并发反而劣化**：c=8 时 ITL 63.6ms、E2E p90 47s，比 base 差很多。可能原因：
  cudagraph 只捕获 `[2,16]` 两个 batch、MTP 在长 prefill+高并发下回退、以及两个实例的
  `gpu-memory-utilization` 不同（0.40 vs 0.30，KV cache 容量不同）；
- **绝对延迟高**（TTFT 1.2~4.8s）是 0.8B 模型 + 单卡 + 9.7k~11k prompt 的真实表现，不是 mock
  的常数延迟；E2E 20~47s 主要被长输出（osl ~250）与排队拉长；
- 单次 120s、共享一张卡、跑在别人的集群上（其他 chip 有负载），数字只用于**验证链路与相对
  趋势**，不要当榜单。

## 256k 上下文：官方语料（独占一张卡）

Qwen3.5-0.8B 的 `max_position_embeddings=262144`。独占 chip 7、`--gpu-memory-utilization 0.85`
起服务后，`--max-model-len 262144` 可用，于是能直接跑**官方 pinned 语料**（不再需要 13k 本地 trace）：

```bash
docker run --rm --net host -v /data/agentx/real_results:/results \
  -v /data/model:/data/model:ro \
  ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64 run \
    --url http://127.0.0.1:8080 --model qwen3.5 \
    --tokenizer /data/model/Qwen3.5-0.8B \
    --target qwen35-256k --concurrency 4 \
    --duration 300 --entries 64 --max-context 250000
```

| 并发 | 请求 | TTFT p50/p90 | ITL p50 | 输出 tok/s | E2E p90 | 缓存（理论/服务端） |
|---|---|---|---|---|---|---|
| 4 | 27 | 7.0 / 9.8 s | 9.5 ms | 35.1 | 35.9 s | 96.1% / 94.2% |
| 8 | 36 | 10.2 / 17.6 s | 44.9 ms | 56.0 | 67.8 s | 92.8% / 93.6% |

说明：300s 窗口里每个 200k 级 prefill 都要十几秒，完成的请求数少；ISL p50 44k 是 AgentX
按轨迹 25%~75% 切片后的窗口值，不是语料被截断。该点用的是官方 `--public-dataset`，只有
时长/条数低于提交阈值（`--unsafe-override`）。

## 复现产物

- `report.html`：镜像里 `agentx-offline report` 生成（ECharts 内联）
- `report_data.json`：报告数据源
- 原始 artifacts 留在服务器 `/data/agentx/real_results/<target>/c<N>/`（未提交，含 7MB
  server_metrics、timeslices、逐请求 jsonl）
