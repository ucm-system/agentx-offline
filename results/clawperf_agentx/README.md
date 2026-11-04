# ClawPerf 能不能回放 AgentX 数据集？

结论：**能回放（缓存/命中率层面），但“结果一样”只对缓存命中率成立；延迟和吞吐不会一样。**

## 1. 格式：能对上

ClawPerf 的 `trace` 模式吃 kvcache.ai JSONL：

```json
{"hash_ids": [0,1,2], "input_length": 192, "block_size": 64, "user_id": "session-x"}
```

AgentX / WEKA 的每条 trace 正好有 `hash_ids`、`in`、`block_size`、子 agent 嵌套，所以写一个
flatten + 按 `t` 排序的转换器即可：

```bash
python scripts/convert_agentx_to_kvcache.py \
  agentx/traces_13k/agentx-syn-000000.json -o /tmp/agentx_one.kvcache.jsonl
# 或用官方语料前 N 条
python scripts/convert_agentx_to_kvcache.py --dataset 8 --hf-home data/hf -o /tmp/agentx8.jsonl
```

转换器把主链请求和 subagent 内层请求都展开成行，`user_id` 区分会话/子 agent，原始 `t` 保留在
`agentx` 字段里（ClawPerf 会忽略它，见下）。

## 2. 缓存命中率：数值完全一致

对 `agentx-syn-000000`（77 个请求、742,774 input tokens、6,180 个块）：

| 指标 | AgentX 理论值（从 trace 直接算） | ClawPerf 模拟（2M token 预算，LRU） |
|---|---|---|
| prefix-cache 命中率 | **98.36%**（11,132 / 11,318 块） | **98.36%**（ceiling 98.36%） |
| unique blocks | 186 | 186 |
| 命中块 token | 11,132 × 64 = 712,448 | 712,448 |

复现：

```bash
clawperf --mode trace --trace-file /tmp/agentx_one.kvcache.jsonl \
  --cache-budget-tokens 2000000 --output results/clawperf_agentx/clawperf_one.json
```

两者算的是同一个东西：块级前缀共享。所以**只看 KV cache 命中率/预算拐点，ClawPerf 可以直接用
AgentX 数据**，数值可复现。

## 3. 真实请求回放：AgentX 数据集缺 `messages`

ClawPerf 的 trace 真实回放要求每行带 `messages`（它会发真实 OpenAI 请求）；AgentX 的语料只有
`hash_ids` + token 数，**没有 prompt 文本**，所以转换出来的 trace 只能做模拟：

```
Real replay: trace has no messages (simulation only).
```

要发真实请求必须自己合成 `messages`（按 hash_ids 拼 token 文本），这会把“同一份语料”变成
“长度相近、内容合成”的负载，已经不能称为同一测试。

## 4. 就算补了 messages，延迟/吞吐也不会一样

| 语义 | AgentX | ClawPerf trace |
|---|---|---|
| 时间轴 | 按 trace `t` 回放，尊重 think time、idle gap（有 cap） | 忽略 `t`，每个会话内请求背靠背发 |
| 并发模型 | 闭环 agent 会话 + subagent DAG（SPAWN/JOIN） | 线性会话；`--trace-users` 只保证同会话顺序 |
| warmup | 10/lane + snapshot primers + cache-bust 首轮前缀 | 无 AgentX 式 warmup/primer |
| 轨迹起点 | 每条 trace 25%~75% 处取快照 | 从 trace 第一条开始 |
| 计数口径 | 服务端 token 数（`use_server_token_count`）、usage.cached_tokens | 客户端 tokenizer 计数 |
| 指标 | AgentX scenario 锁定的 TTFT/ITL/吞吐 | EvalScope 的 TTFT/TPOT + 模拟命中率 |

所以：**同一批 trace，ClawPerf 的命中率能对上，TTFT/吞吐/并发曲线不会对上**；反过来 ClawPerf
的 `scenario`/`slo` 模式又和 AgentX 的轨迹负载不是同一回事。

## 5. 建议怎么用

- 想快速评估「这套 KV cache 配置在 AgentX 流量下的命中率/预算拐点」→ ClawPerf `trace` +
  本转换器，几秒钟出结果，和 AgentX 理论值一致；
- 想复现 AgentX 榜单的 TTFT/吞吐/并发甜点 → 必须用 AgentX harness（本仓库 `agentx-offline`），
  ClawPerf 只能作为旁证；
- 如果把 AgentX 的 `hash_ids` 换成带 `messages` 的自有 trace，ClawPerf 的真实回放才有意义，
  但那时负载已经不是 AgentX 官方语料了。

## 文件

- `agentx_one_trace.kvcache.jsonl`：转换后的单条 trace（77 行）
- `clawperf_one.json` / `clawperf_one.md`：ClawPerf 模拟原始输出
- 转换器：`scripts/convert_agentx_to_kvcache.py`
