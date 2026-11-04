# 上下文窗口 × 并发 × SLA：真实 Qwen3.5-0.8B（910B3 单卡）

回答三个问题：给定硬件下「用户量 → 吞吐」、小上下文窗口怎么跑、以及按 SLA 自动找吞吐最大的并发。

## 1. 窗口怎么选、能跑多少条官方 trace

AgentX 用 `--max-context-length` 过滤峰值 prompt+output 超过窗口的 trace。镜像把它做成
`--window` 标签（留 512 token 余量），官方 393 条语料在各窗口的可用条数：

| 窗口 | 可用 trace | 说明 |
|---|---|---|
| 32K | 0 | 最小一条峰值 ~43k，跑不了 |
| 64K | 9 | 样本很少，并发扫描噪声大 |
| **128K** | **85** | 够做 c=2…32 的扫描 |
| **256K** | **221** | 覆盖 56%，不是“全部” |
| 512K | 323 | |
| 1M | 393 | 全量，需要 ≥1M 窗口的模型 |

命令（服务端 `--max-model-len` 要 ≥ 窗口，否则请求会超长）：

```bash
agentx-offline run --url http://127.0.0.1:8080 --model qwen3.5 \
  --tokenizer /data/model/Qwen3.5-0.8B --target qwen35-128k \
  --window 128k --concurrency 2,4,8,16 --duration 120 --entries 8
```

## 2. 各窗口的实测吞吐（单卡 0.8B，120s/点）

| 窗口 | 并发 | 请求 | TTFT p90 | ITL p50 | 输出 tok/s | SLA 判定 |
|---|---|---|---|---|---|---|
| 64K | 2 | 9 | 13,781 ms | 10.6 ms | 56.4 | 15s 通过 / 5s 失败 |
| 64K | 4 | 20 | 7,196 ms | 10.4 ms | 43.6 | 15s 通过 |
| 64K | 8 | 23 | 6,914 ms | 9.9 ms | 31.3 | 15s 通过 |
| 64K | 16 | 15 | 8,674 ms | 38.4 ms | 63.9 | 失败（成功率 93.3%） |
| 128K | 2 | 8 | 14,280 ms | 7.6 ms | 36.1 | 30s 通过 |
| **128K** | **4** | 17 | 8,812 ms | 21.5 ms | **73.7** | 30s 通过 |
| 256K | 4 | 27 | 9,797 ms | 9.5 ms | 35.1 | 20s 通过 |
| **256K** | **8** | 36 | 17,580 ms | 44.9 ms | **56.0** | 20s 通过 |

注意：64K 只有 9 条 trace，c≥8 时并发会话在复用同一批 trace，数值波动大（c16 反而比 c8 高），
**不要**用 64K 窗口的 c 扫描下“并发越高越好/越差”的结论；它只说明 64K 长 prefill 下单卡吞吐
约 30~65 tok/s。

## 3. 硬件不变时，「用户量 → 吞吐」怎么读

同一台 910B3、同一个 0.8B 模型，窗口越小、可用 trace 越多，吞吐曲线越接近计算上限：

| 负载形态 | 峰值吞吐 | 峰值并发 | 备注 |
|---|---|---|---|
| 13k 合成 trace（8 条、750 请求） | **196 tok/s** | c=16 | 这是“短上下文高并发”的上限；c=32 后回落 |
| 64K 官方子集（9 条） | ~65 tok/s | 不可靠 | 长 prefill 主导，样本太少 |
| 128K 官方子集（85 条） | **73.7 tok/s** | c=4 | c=4 已满足 30s TTFT SLA |
| 256K 官方子集（221 条） | **56 tok/s** | c=8 | 再高 TTFT p90 会破 20s |

结论：**单卡 0.8B 的吞吐不是随用户数线性涨**，长上下文 prefill 很快把算力吃满；256K 下 c=8
左右就是甜点，128K 下 c=4，13k 下 c=16。给 SLA 后“最大吞吐并发”会明显低于“最大吞吐点”。

## 4. SLA 自动调优（`tune`）

```bash
agentx-offline tune --url http://127.0.0.1:8080 --model qwen3.5 \
  --tokenizer /data/model/Qwen3.5-0.8B --target qwen35-128k \
  --window 128k --concurrency 2,4,8,16 --duration 120 --entries 8 \
  --sla-ttft-p90 30000 --sla-itl-p50 50 --sla-min-success 99
```

- 每个点读取官方 `profile_export_aiperf.json` 的 TTFT p90 / ITL p50 / E2E p90 / 成功率；
- 全部满足才算 PASS，推荐 **PASS 集合里 out_tps 最大的并发**；没有 PASS 就退回全局最大并标注；
- 输出 `tune.json` / `tune.md` / `report.html`（含 X/Y 曲线浏览器）；
- 自适应模式：`--auto --start 1 --max-concurrency 128`，每轮翻倍，连续 2 个点 SLA 失败或吞吐
  下滑就停，适合不想手写并发列表时。

本次实测：128K 在 SLA（TTFT p90≤30s、ITL≤50ms、成功率≥99%）下推荐 **c=4（73.7 tok/s）**；
64K 在 5s TTFT SLA 下无解（长 prefill 下 p90 6.9~13.8s），放宽到 15s 后推荐 c=2（56.4 tok/s，
且 c=16 因成功率 93.3% 被淘汰）。

## 5. 复现产物

- `report.html` / `report_data.json`：64K + 128K 两个 target 的对比与曲线浏览器
- `tune_64k.{json,md}` / `tune_128k.{json,md}`：两次 SLA 调优的完整记录
- 256K 数据：`results/real_qwen35/`（官方语料 64 条、c=4/8、300s）
- 13k 高并发扫描：`results/real_qwen35_sweep/`
- 原始 artifacts 在服务器 `/data/agentx/results_windows/<target>/c<N>/`
