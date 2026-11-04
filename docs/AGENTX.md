# AgentX 调研报告：SemiAnalysis InferenceX 的 agentic 推理 benchmark

> 版本锚点：`agentx-harness==1.0.7`（wheel 内 `aiperf` = NVIDIA AIPerf 上游 + SemiAnalysis scenario 锁）。
> 本文所有结论均可由 `%TEMP%\agentxw\x\` 下解包出的源码复核，关键文件路径在每节末尾给出。
> 目标读者：要在 `GPT-Zero` mock 服务上**离线复现** AgentX 负载形状的工程师。

---

## 0. TL;DR（先看这 6 条）

1. **AgentX 不是"Synthetic LLM benchmark"，是 trace 回放 benchmark**：它把 SemiAnalysis 从生产 Claude Code 会话里抓的 WEKA trace（393 条）按原始 `t` 时间轴重放，测的是**整机/整集群在真实 agentic 时序下的吞吐与交互性**，不是单请求延迟。
2. **harness 锁死了一批参数**（`scenario` 机制）：`timing_mode=agentic_replay`、必须 `ignore_eos`、必须 `stream`、禁止 `--ignore-trace-delays`、禁止输入截断、benchmark 时长下限 900s、`cache_bust=FIRST_TURN_PREFIX`。这些不是建议值，是 validator 会**改写或拒绝**的硬约束。
3. **负载形状极端偏缓存**：median 输入 **142k tokens**、median 输出 **444 tokens**、**98% 的输入 token 来自 prefix cache 命中**。所以 `prefix_cache_hits_total` 和 TTFT 才是主角，decode 吞吐不是。
4. **`t` 的单位是秒，不是毫秒**。`WekaNormalRequest.t: float` 注释为 "Request timestamp in seconds"；loader 内部 `req.t * 1000.0` 转 ms 后喂给 AIPerf 的 `Turn.timestamp`。
5. **闭环并发语义**：并发 N = N 条 trajectory lane，每 lane 绑定一条 trace，第 k 个请求在 `t_k`（相对 trace 起点）发出，但**同一流内必须等前一个请求结束**（delay 是 end-to-start 的 idle gap，`max(0, t_k - (t_{k-1}+api_{k-1}))`），子代理 SPAWN 后父链的 JOIN turn 要等子代理结束。
6. **SPEC/accuracy 与 AgentX 无关**：官方明确 AgentX 只测性能，不测正确性（见 §9）。

---

## 1. AgentX 是什么

| 项 | 内容 |
|---|---|
| 官方页面 | <https://inferencex.semianalysis.com/agentx> |
| 快速上手 | <https://inferencex.semianalysis.com/agentx/quickstart> |
| 方法论 | <https://inferencex.semianalysis.com/agentx/methodology> |
| 发行包 | PyPI `agentx-harness==1.0.7`（提供 `agentx` 命令行） |
| 上游 | NVIDIA **AIPerf**（`aiperf` 包），AgentX 只加一层「品牌 + scenario 锁」 |
| 定位 | 用**真实 Claude Code 会话 trace** 回放 agentic 推理负载，出 `throughput/chip`、TTFT、interactivity 等指标，用于跨硬件/推理栈的横向对比 |

AgentX 与 AIPerf 的关系，一句话：**`agentx` = `aiperf` + `--scenario agentx` 默认注入**。

```python
# agentx_cli.py（wheel 内，1900 字节全文）
def main(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "profile":
        options = tokens[1 : tokens.index("--") if "--" in tokens else len(tokens)]
        has_scenario_or_config = any(
            token in {"--scenario", "--config", "-f"}
            or token.startswith(("--scenario=", "--config=", "-f"))
            for token in options
        )
        if not has_scenario_or_config:
            tokens[1:1] = ["--scenario", "agentx"]   # ← 唯一实质动作
    app(tokens)
```

所以：**只要不带 `--scenario` / `--config`，所有 `agentx profile` 都跑 `agentx` scenario**；显式 `--scenario inferencex-agentx-mvp` 可退回 MVP 档。

---

## 2. Scenario 锁：官网 benchmark 到底用什么参数跑

源码：`aiperf/common/scenario/inferencex_agentx_mvp.py`（全文 70 行，下方为等价的参数表）。
两个 spec 是**继承关系**：`agentx` = `inferencex-agentx-mvp` + 一组 override。

### 2.1 `inferencex-agentx-mvp`（MVP 基线）

| 参数 | 值 | 含义 / 影响 |
|---|---|---|
| `timing_mode` | `agentic_replay` | 按 trace 的 `t` 时间轴发请求（不是 fixed_schedule / concurrency 相位） |
| `require_ignore_eos` | `True` | 必须发 `ignore_eos=true`，服务端**精确产出 `out` 个 token**，否则 TTFT/吞吐不可比 |
| `require_streaming` | `True` | 必须 SSE 流式；非流式拿不到 TTFT |
| `forbid_ignore_trace_delays` | `True` | 禁止 `--ignore-trace-delays`：trace 里的 idle gap 必须真实等待，否则并发被人为拉高 |
| `forbid_input_truncation` | `True` | 禁止 `--input-truncation`：142k 的 prompt 不许截断成 4k |
| `min_benchmark_duration_seconds` | `900` | 低于 15min 直接拒绝（trace 覆盖不足，噪声大） |
| `default_benchmark_duration_seconds` | `1800` | 默认 30min |
| `default_trajectory_start_min_ratio` | `0.0` | 轨迹起点可落在 trace 的 0% |
| `default_trajectory_start_max_ratio` | `1.0` | …到 100%（全区间随机取一段） |
| `system_idle_gap_cap_seconds` | `10.0` | **系统级**空闲 gap 上限：trace 里 >10s 的静默被夹到 10s（否则一次 coffee break 会污染统计） |
| `forbid_inter_turn_delay_cap` | `True` | 禁止 `--inter-turn-delay-cap`（除了上面的 system cap，不许再压 delay） |
| `require_cache_bust` | `FIRST_TURN_PREFIX` | 每条 trace 首轮 prompt 前**必须**插 `[rid:xxxxxxxxxxxx]` 标记（见 §6） |
| `minimum_profile_metric_coverage_ratio` | `0.95` | 95% 的请求必须采到完整指标，否则报告不生成 |
| `require_loader` | 见右 | `semianalysis_cc_traces_weka_with_subagents*`（060226/060526/060826/061326/061526/062126 及 `_256k` 变体）+ 文件版 `weka_trace` / `weka_hf`。**只允许这些 loader**，自造格式要在名单内才跑得起来 |

### 2.2 `agentx`（`agentx profile` 默认注入）

在 MVP 之上 override：

| 参数 | 值 | 含义 / 影响 |
|---|---|---|
| `default_benchmark_duration_seconds` | `3600` | 默认 **1 小时**（不是 30min） |
| `default_trajectory_start_min_ratio` | `0.25` | 起点落在 25%~75%，避开"会话刚开始"的空窗与尾部 |
| `default_trajectory_start_max_ratio` | `0.75` | 同上 |
| `cli_defaults.endpoint_type` | `chat` | 走 `/v1/chat/completions`，不是 `/v1/completions` |
| `cli_defaults.streaming` | `True` | |
| `cli_defaults.stats_interval` | `30` | 30s 打一次中间统计 |
| `cli_defaults.random_seed` | `42` | 轨迹切分/采样可复现 |
| `cli_defaults.failed_request_threshold` | `0.10` | 失败率 >10% 判整轮 invalid（不静默出数） |
| `cli_defaults.warmup_requests_per_lane` | `10` | 每 lane 额外 10 条 warmup（叠加 §7 的 mandatory primers） |
| `cli_defaults.warmup_grace_period` | `1800` | warmup 允许慢 30min（142k prefill 本身就很慢，不是卡死） |
| `cli_defaults.trace_idle_gap_cap_seconds` | `300` | **单 trace 内** gap 上限 300s（比 system 级 10s 宽，用于跨子代理长思考） |
| `cli_defaults.use_server_token_count` | `True` | 用服务端返回的 `usage` 计 token，不用本地 tokenizer 估算 |
| `cli_defaults.no_gpu_telemetry` | `True` | 不采 GPU 遥测（纯 HTTP 负载侧指标） |
| `cli_defaults.conversation_num_dataset_entries` | `393` | 数据集条目数 = **393 条 trace** |
| `cli_defaults.slice_duration` | `1.0` | 1s 切片做时序对齐/画图 |
| `environment_defaults.DATASET.CONFIGURATION_TIMEOUT` | `1800` | 数据集构建超时 |
| `environment_defaults.SERVICE.PROFILE_CONFIGURE_TIMEOUT` | `1800` | 服务 profile 探测超时 |
| `environment_defaults.UI.REALTIME_METRICS_ENABLED` | `True` | |
| `environment_defaults.HTTP.TCP_USER_TIMEOUT` | `900000` ms | 15min TCP user timeout（防长 prefill 被内核掐断） |

### 2.3 等价手工命令（可复制）

```bash
# 官网等价跑法（默认注入 scenario agentx）
agentx profile \
  --url http://<endpoint>:8000 \
  --endpoint-type chat --streaming \
  --input-file <traces_dir> --custom-dataset-type weka_trace \
  --unsafe-override \
  --tokenizer deepseek_v3_tokenizer --model <served_model> \
  --concurrency 8 --benchmark-duration 3600
```

要点：`--unsafe-override` 是**必需**的 —— 没有它，validator 会对上面那批 `require_*` 违约项报错退出；有了它，违约项被"记录并接受"（provenance 里留痕），这正是本地小规模冒烟（`--benchmark-duration 30`）能跑起来的原因。

---

## 3. 数据集

| 项 | 值 |
|---|---|
| Registry 名 | `semianalysis_cc_traces_weka_062126`（当前 AgentX 默认语料）/ `..._062126_256k` |
| HF repo | `semianalysisai/cc-traces-weka-*`（日期钉版本：`062126`、`061526`、`061326`、`060826`、`060526`、`060226`） |
| 条数 | **393** 条 trace（`conversation_num_dataset_entries=393`） |
| 来源 | 生产环境 Claude Code 会话（CC ≥ 2.1.139，v5-only），含 Task 工具派生的**子代理扇出** |
| `256k` 变体 | 同一语料按 256k 上下文窗口过滤/裁剪，用于 256k 服务档；非 256k 变体面向 128k 档 |
| `_no_subagents` 变体 | `cc-traces-weka-no-subagents-051826`：98 条，剥掉 subagent 块，只留主链（旧默认，现为别名） |

**形状统计（官方报告口径）**：

- 输入 median **142k tokens**，p 尾部可达 ~200k+（这就是为什么禁止 input truncation，也是为什么 TTFT 方差极大）
- 输出 median **~444 tokens**（agentic 场景输出短：一句话 + 一次 tool call）
- **98% 的输入 token 是 cached prefix**（同一会话反复携带同一份 system prompt + 工具定义 + 历史）
- 因此：**缓存命中率是 AgentX 的第一性指标**，HBM 带宽/L2 不是瓶颈，KV cache 容量与 prefix cache 效率才是

一条 trace = 一个会话（`id` = session id），内部是主链请求 + 若干 `subagent` 块的**扁平交错列表**。

---

## 4. WEKA trace 格式（严格 schema）

源码：`aiperf/dataset/loader/weka_trace_models.py`（pydantic，`extra="forbid"`，即**多一个字段就报错**）。

### 4.1 顶层

```json
{
  "id": "trace-000001",
  "models": ["claude-sonnet-4-5"],
  "block_size": 64,
  "hash_id_scope": "local",
  "tool_tokens": 12000,
  "system_tokens": 8000,
  "requests": [ ... ],
  "totals": { "optional": "opaque" }
}
```

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `id` | str | 必填，**全语料唯一** | 会话 ID；重复会让 loader 抛 `Duplicate trace id` |
| `models` | list[str] | 必填 | trace 内用到的模型（映射到请求的 `model` 字段） |
| `block_size` | int | `> 0` | KV cache 块大小（token），真实抓取里是 64 |
| `hash_id_scope` | `"local"` | **只能是 `local`** | `global`（跨 trace 共享 KV）在 schema 层就被拒 |
| `tool_tokens` | int | 默认 0 | 工具定义前缀的 token 数（属于可缓存前缀） |
| `system_tokens` | int | 默认 0 | system prompt 前缀的 token 数 |
| `requests` | list | 必填 | 交错的 n / s / subagent 列表 |
| `totals` | dict\|null | 可选、不透明 | trace 级汇总，loader 不解释 |

### 4.2 `type: "n"` 普通请求 / `type: "s"` 流式请求

```json
{
  "t": 12.345,
  "type": "s",
  "model": "claude-sonnet-4-5",
  "in": 145233,
  "out": 312,
  "hash_ids": [0, 1, 2, 3, ...],
  "input_types": ["text"],
  "output_types": ["tool_use"],
  "stop": "tool_use",
  "api_time": 8.42,
  "think_time": 0.31,
  "ttft": 1.02
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `t` | float | ✅ | **请求时间戳，单位「秒」，相对会话起点**（⚠️ 不是毫秒。loader `req.t * 1000.0` 转 ms） |
| `type` | `"n"` \| `"s"` | ✅ | 判别器：n=普通（非流式），s=流式 |
| `model` | str | ✅ | 该请求的模型标识 |
| `in` | int | ✅ | 输入 token 数（alias `input_length`） |
| `out` | int | ✅ | 期望输出 token 数（alias `output_length`）→ 回放时作 `max_tokens`；0 会被抬到 1 |
| `hash_ids` | list[int] | 默认 `[]` | KV cache 块 ID（**每块 = `block_size` 个 token**） |
| `input_types` | list[str] | 默认 `[]` | 内容类型标注（`text`/`image`/…），回放不强依赖 |
| `output_types` | list[str] | 默认 `[]` | 输出类型标注 |
| `stop` | str | 默认 `""` | 抓取时的停止原因：`""`/`tool_use`/`end_turn`（仅记录，不回放） |
| `api_time` | float\|null | 可选 | 抓取时服务端处理耗时（秒）。**回放用不到**（会由被测服务真实耗时替代），但用于 JOIN 判定与 delay 折算 |
| `think_time` | float\|null | 可选 | 用户在两次请求之间的思考时间（秒） |
| `ttft` | float\|null | 可选（仅 `s`） | 抓取时的 TTFT（秒），仅记录 |

> ⚠️ 任务书里写的「`t` 毫秒」与 wheel 内 schema 不一致。**以 wheel 为准：`t` 是秒。**
> 本仓库 `agentx/trace_gen.py` 生成的也是秒；`validate_trace()` 会检查单调性与量级。

### 4.3 `type: "subagent"` 子代理块

```json
{
  "t": 40.0,
  "type": "subagent",
  "agent_id": "agent_001",
  "subagent_type": "Explore",
  "duration_ms": 18250,
  "total_tokens": 23400,
  "tool_use_count": 7,
  "status": "completed",
  "requests": [ { "t": 41.2, "type": "s", ... } ],
  "models": ["claude-sonnet-4-5"],
  "tool_tokens": 12000,
  "system_tokens": 8000
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `t` | float | **SPAWN 时刻**（秒，根 trace 坐标系） |
| `agent_id` | str | 子代理 ID，如 `agent_001` |
| `subagent_type` | str | 类型（`Explore` 等），只记录 |
| `duration_ms` | int\|null | 墙钟时长（ms）。`status='async_launched'` 时为 `null` |
| `total_tokens` / `tool_use_count` | int\|null | `async_launched` 时 `null` |
| `status` | str | `completed` 或其他终态 |
| `requests` | list[n\|s] | 子代理自己的请求链（nested） |
| `models` | list[str] | 子代理用的模型 |
| `tool_tokens` / `system_tokens` | int | 子代理自己的前缀 token 数 |

**子代理内层 `t` 的两种写法 loader 都吃**：绝对时间戳（现代抓取）或相对 spawn 的相对时间（老 fixture）。判定逻辑：`req.t < entry.t` 视为相对，自动 `+ entry.t`。

### 4.4 JOIN 语义

> "The parent's next `WekaNormalRequest` in the outer list is understood to occur after this subagent completes (spec §4.4)."

即：subagent 块之后的**第一条父级 n/s 请求**是 JOIN 点，它必须等 `sa_end = entry.t + duration_ms/1000` 之后才开始（`duration_ms` 缺失时退化为 `max(inner.t + inner.api_time)`）。这就是「SPAWN/JOIN 门控」。

### 4.5 hash_ids 的真实语义（prefix cache 建模核心）

`hash_ids` 不是"token id"，是**块级 KV cache 指纹**：第 i 个 hash 对应一个 `block_size` token 的块。回放时用 hash 重现 prompt 内容，从而让服务端 prefix cache 自然命中：

- `input_length == len(hash_ids) * block_size`：**精确铺满**，每块完整
- `(M-1)*block_size < input_length < M*block_size`：**末块部分**（AIPerf 自造 prompt 用）
- `input_length > M*block_size`：**仅前缀**（真实抓取形态）——已缓存 M 块 + 一段未缓存的 sha256 采样尾部（新 token）

服务端因此能算出：命中块数 = 与前序请求 `hash_ids` 的**最长公共前缀**（LCP）。同一 session 内 hash 局部递增，跨请求共享前缀 → 天然 98% 命中。

---

## 5. 时间语义与闭环并发

### 5.1 三层时间缩放

| 层 | 规则 | 代码位置 |
|---|---|---|
| trace 内 gap | `system_idle_gap_cap_seconds=10.0`（系统级强制）/ `trace_idle_gap_cap_seconds=300`（单 trace 放宽） | `scenario/base.py`, `weka_trace.py` |
| 请求间隔 | **end-to-start idle gap**：`delay_k = max(0, t_k - (t_{k-1} + api_{k-1}))`，非 start-to-start | `_end_to_start_delay_ms()` |
| 轨迹切片 | 起点 `t*` 落在 trace 的 `[0.25, 0.75]` 比例处（按 turn 百分比），`slice_duration=1.0s` 对齐 | `timing/trajectory_source.py` |

最后一条很关键：**AgentX 不是从头重放**，它在每条 trace 的时间轴 25%~75% 处取一个快照点 `t*`，从那里开始跑。这样：
- 避开冷启动（0% 处系统 prompt 还没热）和长尾（75% 之后会话快结束、并发塌陷）
- 每个 lane 的快照点独立采样，所以 393 条 trace 支撑 N 个 lane 时会有复用（`concurrency > pool` 时同一 trace 在不同 lane 以不同 `t*` 出现）

### 5.2 并发 = lane 数

`--concurrency N` = N 条**独立 trajectory lane**，每 lane 独占一条 trace、按 `t` 调度、**流内串行**（前一个请求返回后才可能发下一个）。因此：

- 系统实际在飞请求数 ≤ N，且**远小于断面理论值**（因为 98% 时间在等 prefill 完成 → 强相关的"脉冲式"并发）
- 这就是为什么 AgentX 的 throughput/chip 曲线与 `bench` 的固定并发压测曲线**形态不同**：AgentX 是 open-loop 时序驱动的 closed-loop 并发

### 5.3 子代理并发

`subagent` 块在 `t` 时刻 SPAWN 出**一条独立的 child conversation**（有自己的 session id 后缀，如 `trace-1::sa:agent_001`），它与父链**并行**跑；父链的下一条请求是 JOIN 点，被 `sa_end` 门控。因此一条 trace 在飞请求数 > 1，峰值可达 `1 + 子代理数`。

---

## 6. cache-bust marker 规则

源码：`aiperf/timing/strategies/cache_bust.py`

```
digest = sha256(f"{benchmark_id}:{recycle_pass}:{trajectory_index}:{trace_id}").hexdigest()[:12]
rid    = f"[rid:{digest}]"
target == FIRST_TURN_PREFIX  →  marker = f"{rid}\n\n"    # 放在首轮 prompt 的【最前面】
target == TURN_SUFFIX        →  marker = f"\n\n{rid}"    # 放在最后
```

要点：
- **同一条 trace 树的所有成员（主链、子代理、flat agent）共用同一个 marker**（按 root trace id 派生），保证 warmup → profiling 的 KV 复用不被 marker 打断
- `recycle_pass` 每条 trace 独立计数：一条 trace 被第二个 lane 复用时 `recycle_pass+1` → **不同 digest** → 强行让服务端 prefix cache 冷启动，防止跨 lane "偷热"
- 12 hex = 48 bit，足够一轮内唯一

**为什么要它**：393 条 trace 的 system prompt / 工具定义高度模板化，如果不打断，第二次跑的 provider 会白拿一次"全命中"的假好成绩。AgentX 必须在**每次新 trajectory tree 的首轮**注入非缓存前缀。

---

## 7. warmup / primer

两段机制叠加（`timing/trajectory_source.py::warmup_credit_counts_by_lane`）：

1. **mandatory snapshot primers（强制的）**：由 `t*` 快照点决定 —— 在 `t*` 之前已经发出过请求的每个流（含正被 JOIN 门控的父链），都要把**它 `t*` 前的最后一个 turn** 重放一次（`warmup_turn_index`），用来把该 lane 的 prefix cache 预热到"跟线上一样热"。没有快照/没算出来的 lane 至少补 1 条。
2. **`warmup_requests_per_lane=10`（可配的）**：在 primer 之上再加 10 条/ lane。总 warmup 预算 = `sum(primers) + 10 * lanes`。

`warmup_grace_period=1800` 允许 warmup 最长 30min —— 因为 primer 是 142k token 的 prefill，本来就慢。warmup 阶段若出现终态失败且超过阈值，抛 `TrajectoryWarmupFailedError`，**profiling 阶段不会启动**（不诚实地出数）。

**Warmup 数据不进统计**：`minimum_profile_metric_coverage_ratio=0.95` 只对 profiling 阶段生效。

---

## 8. 官方报告指标

| 指标 | 定义 | 依赖 |
|---|---|---|
| **Throughput / chip** | 总输出 token 数（或总 token 数）÷ 墙钟时长 ÷ 芯片数 | 需要知道设备芯片数；本地 mock 无此概念 → 可空 |
| **TTFT（p50/p90/p99）** | 首 token 延迟；142k prefill 下是主战场 | SSE 首帧 + 流式 |
| **TPOT / ITL** | 每输出 token 时间 / token 间隔 | 需要 `ignore_eos` 保证 token 数确定 |
| **Interactivity / p90** | 交互性：`1 / p90(TPOT)` 一类的人机可感速率；官方按 p90 口径给"每秒可感知 token" | 分位统计 |
| **Request throughput** | req/s（受 trace 时序约束，波动大） | |
| **Cache hit rate** | `prefix_cache_hits_total / prefix_cache_queries_total` | 服务端 Prometheus |
| **Success rate / failed_request_threshold** | 失败率 >10% → 整轮 invalid | |
| **Concurrency profile** | 在飞请求数随时间（1s 切片）曲线 | `slice_duration=1.0` |

注意：AgentX 报告**不给 accuracy / SPEC / pass@k** —— 见下节。

---

## 9. SPEC 与 quality 无关

官方 methodology 明确：AgentX 是**性能** benchmark。
- **不测**模型回答质量、代码正确率、tool call 正确性（这些由 SWE-bench / τ-bench 之类回答）
- **不考虑**投机解码 acceptance rate（speculative decoding 的收益在此负载下由 `ignore_eos` 固定 token 数间接体现，但 harness 不度量 acceptance）
- 唯一"正确性"约束是**形状正确**：token 数对得上（`ignore_eos`）、流式对得上、时长够、失败率达标

因此本地复现时：mock 的"回答内容"可以任意，但**token 计数、流式时序、缓存命中语义必须对**。

---

## 10. 「AgentX 参数 → 本 mock 服务如何复现」映射表

> 左列 = §2 的 scenario 参数；中列 = 在 `GPT-Zero` mock 上等价做什么；右列 = 复现程度。

| AgentX 参数 / 行为 | 本仓库复现方式 | 程度 |
|---|---|---|
| `timing_mode=agentic_replay` | `agentx/replay.py`：按 trace `t` 调度 + 流内串行 + 子代理并行 + JOIN 门控 | ✅ 完全 |
| `require_ignore_eos=True` | 请求带 `ignore_eos=true`，服务端精确产出 `max_tokens` 个 token（CONTRACT §3.1） | ✅ 完全 |
| `require_streaming=True` | `stream=true` + `stream_options.include_usage=true`，逐 chunk 计 TTFT/ITL | ✅ 完全 |
| `forbid_ignore_trace_delays` | 回放器**不提供** `--ignore-delays` 开关 | ✅ 完全 |
| `forbid_input_truncation` | 生成器 `--max-isl` 只限制**生成**规模，回放不做任何截断 | ✅ 完全 |
| `min/default_benchmark_duration` | `--duration` 参数，本地默认短（30~120s），文档标注与 900s 的差距 | ⚠️ 缩短 |
| `trajectory_start_range 0.25~0.75` | `--start-ratio-min/--start-ratio-max`（默认 0.25/0.75），按 turn 比例取快照 | ✅ 完全 |
| `system_idle_gap_cap_seconds=10.0` | `--system-idle-gap-cap 10.0`：对 `t` 序列做 gap 夹紧 | ✅ 完全 |
| `trace_idle_gap_cap_seconds=300` | `--trace-idle-gap-cap 300` | ✅ 完全 |
| `forbid_inter_turn_delay_cap` | 不实现额外 cap 开关 | ✅ 完全 |
| `cache_bust=FIRST_TURN_PREFIX` | `[rid:{sha256(...)[:12]}]\n\n` 前缀，digest 元组同构（`benchmark_id:pass:trajectory_index:trace_id`） | ✅ 完全（同构） |
| `hash_ids` + `block_size` | 生成器造块 ID；回放器用共享前缀造 prompt，让服务端 prefix block cache 命中 | ✅ 同构 |
| 98% cached / 142k in / 444 out | `--max-isl 142000 --out-median 444`；生成器按会话共享前缀让命中率落在 ~95%+ | ⚠️ 统计近似 |
| `warmup_requests_per_lane=10` + primers | `--warmup 10` + `--primer`（每条 lane 重放 `t*` 前最后一个 turn） | ✅ 完全 |
| 子代理 SPAWN/JOIN | `--subagents 0.4` 生成 + 回放器并发调度 + `duration_ms` 门控 | ✅ 完全 |
| `/metrics` prefix cache 差值 | 读 `vllm:prefix_cache_hits_total` / `_queries_total` 前后差（CONTRACT §3.6） | ✅ 完全 |
| `use_server_token_count` | 用响应 `usage` 计 token | ✅ 完全 |
| `failed_request_threshold=0.10` | 失败率 >10% 时结果标 `valid=false` | ✅ 完全 |
| `conversation_num_dataset_entries=393` | 默认语料条数（生成器可配） | ⚠️ 合成语料 |
| `throughput/chip` | `--chips N`（可选）；缺省输出空值 | ⚠️ 需人工提供 |
| 真实模型质量 | **不可复现**（mock 无模型） | ❌ |
| HF 数据集下载 `semianalysisai/cc-traces-weka-*` | **不可复现**（离线环境无 HF） | ❌ |
| 厂商 chat template / tokenizer 差异 | **不可复现**（用本地 `deepseek_v3_tokenizer` 或 heuristics） | ❌ |
| speculative acceptance / GPU 遥测 | **不可复现**（`no_gpu_telemetry=True`，且 mock 无 GPU） | ❌ |
| 真实 KV cache 容量/驱逐 | 弱复现（mock 的 `--cache-blocks` LRU 只是近似） | ⚠️ |

---

## 11. 本仓库实现（`agentx/`）

```
agentx/
  __init__.py
  __main__.py      # python -m agentx ...
  cli.py           # generate | replay | report | validate
  trace_gen.py     # WEKA trace 合成器 + validate_trace()
  replay.py        # 闭环并发回放器 + 指标 + 图表
  traces/          # 生成的 trace 语料（json，每条一个文件）
```

### 11.1 生成

```bash
.venv\Scripts\python.exe -m agentx generate ^
  --sessions 8 --requests 30 --subagents 0.4 ^
  --max-isl 32000 --out agentx\traces
```

### 11.2 校验

```bash
.venv\Scripts\python.exe -m agentx validate --traces agentx\traces
```

### 11.3 回放（需 mock 服务在跑）

> 端口约定（本机）：`8080` 被 piforged 常驻占用，**团队统一用 8090**（`bench` 的 sweep 也用它）；
> 想隔离自测可另起 8092。`--url` 可指向任何端口，`8080` 只是历史默认。

```bash
# 终端 A：起 mock（团队 8090；自测可换 8092）
.venv\Scripts\python.exe server.py --port 8090 --tokenizer deepseek_v3_tokenizer ^
  --ttft-ms 120 --tpot-ms 8 --max-concurrency 64

# 终端 B：回放（并发 1/2/4，每点 30s，warmup 10/lane + primer）
.venv\Scripts\python.exe -m agentx replay ^
  --traces agentx\traces --url http://127.0.0.1:8090 ^
  --model gpt-3.5-turbo --concurrency 1,2,4 --duration 30 ^
  --warmup 10 --out results\agentx
```

产出：`results/agentx/replay.{json,md,csv}` + `results/agentx/charts/*.png|svg`（吞吐/TTFT vs 并发、缓存命中率 vs 并发）。
实测数值与复现记录见 `docs/AGENTX_REPRO.md`。

### 11.4 报告

```bash
.venv\Scripts\python.exe -m agentx report --json results\agentx\replay.json --out results\agentx
```

---

## 12. 源码索引（复核用）

| 主题 | wheel 内路径 |
|---|---|
| scenario 定义 | `aiperf/common/scenario/inferencex_agentx_mvp.py` |
| validator / 参数改写 | `aiperf/common/scenario/validator.py` |
| trace schema | `aiperf/dataset/loader/weka_trace_models.py` |
| 回放重建（SPAWN/JOIN/链检测） | `aiperf/dataset/loader/weka_trace.py`（2562 行） |
| hash_ids → prompt | `aiperf/dataset/loader/hash_ids_synthesis.py`、`weka_prompt_compose.py`、`aiperf/dataset/generator/prompt.py::_build_token_sequence` |
| cache-bust marker | `aiperf/timing/strategies/cache_bust.py` |
| 轨迹切片 / warmup primer | `aiperf/timing/trajectory_source.py` |
| HF 语料加载 | `aiperf/dataset/loader/semianalysis_cc_traces_weka.py` |
| CLI 品牌与默认 scenario | `agentx_cli.py` |

参考链接：
- <https://inferencex.semianalysis.com/agentx>
- <https://inferencex.semianalysis.com/agentx/quickstart>
- <https://inferencex.semianalysis.com/agentx/methodology>
- <https://raw.githubusercontent.com/ai-dynamo/aiperf/main/docs/tutorials/weka-trace.md>

---

## 13. 已知偏差（诚实清单）

1. **`t` 单位**：任务书说毫秒，wheel schema 是秒。本文与代码统一用**秒**。
2. **真 harness 冒烟**：`experiments/agentx_harness/` 已装 `agentx-harness==1.0.7` 并实跑 ——
   scenario 锁被真实注入（`timing_mode=agentic_replay` / `ignore_eos=true` /
   `cache-bust=first_turn_prefix` / `system-idle-gap-cap-seconds=10.0`），但卡在
   tokenizer/dataset 配置阶段。完整证据链、卡点与绕过方式见 `docs/AGENTX_REPRO.md`。
3. **规模**：本地默认 8 session × 30 请求，比 393 × 数百请求小 3 个数量级；形状对齐，绝对数值不可与官网对比。
4. **throughput/chip** 无真实芯片数，默认留空。
