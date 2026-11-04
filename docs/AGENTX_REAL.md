# AgentX 真机跑通记录（真 harness + 真语料 + 离线）

> 这份文档记录的是**用官方 `agentx-harness==1.0.7` 真跑**的结果，不是本仓库自研回放。
> 服务端是 `gptzero` mock；语料是 SemiAnalysis 公开的 AgentX 语料；全程**断网可复现**。

## 0. 一句话结论

**官方 harness 可以在离线环境端到端跑通**：加载真实 AgentX 语料（393 条中的 8 条）、按官方
scenario 锁（`agentic_replay`/`ignore_eos`/`streaming`/`cache_bust=first_turn_prefix`）回放，
产出官方格式的 `profile_export_aiperf.json`、`profile_export.jsonl`、`server_metrics_export.json`
与 1s 时序切片（timeslices）。分数不能与官网横比（mock 无模型、单机、时长被压缩）。

## 1. 离线准备（只做一次）

```powershell
# 1) HF 镜像（huggingface.co 的 resolve 端点在本地会被 reset，hf-mirror.com 可用）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME     = "D:\Vibe_Workspace\GPT-Zero\data\hf"   # 已在 .gitignore

# 2) 预下载真实语料并生成 Arrow 缓存（1.85 GB JSONL -> 4 shards）
.venv-agentx\Scripts\python.exe experiments\agentx_real\prefetch_dataset.py

# 3) 预下载 tokenizer（DeepSeek-V3 与 mock 内置 deepseek_v3_tokenizer 同源）
.venv-agentx\Scripts\python.exe -c "from aiperf.common.tokenizer import Tokenizer; Tokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')"
```

之后的所有运行都带：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HOME = "D:\Vibe_Workspace\GPT-Zero\data\hf"
```

## 2. 跑通前踩的 5 个坑（都已修，供复现）

| # | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | tokenizer 配置阶段进程直接消失（无异常） | `sentencepiece==0.2.2` 在本机 DLL `access violation` | 降到 `sentencepiece==0.2.0` |
| 2 | 启动即 `Failed to initialize ZMQ socket: Address in use (47xxx)` | AIPerf 在 Windows 上用「TCP loopback 替代 ipc://」，端口从 `28000 + sha256 % 20000` 随机取，和本机环境冲突 | `AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT=20000`、`AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE=6000` |
| 3 | `OSL mismatch: got 256, requested 4045` | AIPerf 发的是 OpenAI 新字段 `max_completion_tokens`，旧 mock 只认 `max_tokens` | mock 增加别名（`gptzero/app.py`） |
| 4 | AgentX 警告「no prompt-cache read tokens were seen」 | OpenAI/vLLM 会在 `usage.prompt_tokens_details.cached_tokens` 报告缓存命中，mock 没给 | mock 的 usage 增加该字段 |
| 5 | 393 条 trace 重建时内存打满（31GB 机器 free→0） | AIPerf 会并行重建全部会话（16 worker） | 离线小机器加 `--num-dataset-entries 8`（仍是真实语料的前 8 条） |

## 3. 实际命令（可直接复制）

```powershell
# mock（另一个终端；单进程，纯离线）
.venv\Scripts\python.exe server.py --port 8092 --tokenizer deepseek_v3_tokenizer `
  --ttft-ms 120 --tpot-ms 8 --max-concurrency 64 --cache-blocks 8192

# 真 harness：8 条真实 trace，4 个并发 agent 会话，180s
$env:HF_HOME="D:\Vibe_Workspace\GPT-Zero\data\hf"; $env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"; $env:HF_DATASETS_OFFLINE="1"
$env:AIPERF_SERVICE_WINDOWS_TCP_BASE_PORT="20000"
$env:AIPERF_SERVICE_WINDOWS_TCP_PORT_RANGE="6000"

.venv-agentx\Scripts\agentx.exe profile `
  --url http://127.0.0.1:8092 --endpoint-type chat --streaming `
  --public-dataset semianalysis_cc_traces_weka_062126 `
  --tokenizer deepseek-ai/DeepSeek-V3 --model gpt-3.5-turbo `
  --concurrency 4 --benchmark-duration 180 --num-dataset-entries 8 `
  --stats-interval 30 --unsafe-override `
  --output-artifact-dir experiments\agentx_real\run_c4_180
```

`--unsafe-override` 只 downgrade 两条合规项：本地时长 <900s、以及（可选）本地语料非 pinned 别名；
本命令用的是 `--public-dataset` pinned 别名，只触发时长一条。

## 4. 结果（`results/agentx_real/`）

运行：2026-10-28 18:02–18:06（Asia/Shanghai），mock = `gptzero` v3，真实语料 8 条，
`--concurrency 4`，profiling 180s，warmup = primer + 10/lane，58 个请求全部成功。

| 指标 | 值 |
|---|---|
| 请求数 / 成功率 | 58 / 100% |
| RPS | 0.28 |
| TTFT p50 / p90 / p99 | **160 / 239 / 370 ms** |
| ITL p50 | 8.7 ms |
| E2E p50 / p90 | 1175 / 2491 ms |
| 输出吞吐 | 103 tok/s |
| ISL p50 | 54,258 tokens（真实 AgentX 语料的长上下文） |
| OSL p50 | 124 tokens |
| 前缀缓存命中 | 理论 **95.5%** / 服务端实测 **93.4%** |

![TTFT 时序](../results/agentx_real/charts/timeline_ttft.png)

（图：`results/agentx_real/charts/timeline_ttft.{png,svg}`、`timeline_throughput.{png,svg}`；
表：`results/agentx_real/summary.{md,csv,json}`。）

## 5. 官方产物（可直接进 AgentX 分析/画图流程）

```
experiments/agentx_real/run_c4_180/
  profile_export_aiperf.json          聚合指标（TTFT/ITL/E2E/吞吐/ISL/OSL…）
  profile_export_aiperf.csv
  profile_export.jsonl                逐请求记录（58 条）
  profile_export_aiperf_timeslices.csv/.json   1s 切片（51 个切片）
  server_metrics_export.json/.csv     mock 的 Prometheus 全量快照
  profile_export_console.txt          官方 console 报告（含 Theoretical Prefix Cache Hit）
  logs/aiperf.log                     全量日志（含 scenario 注入与每条实时指标）
```

## 6. 稳定性与已知问题

- **c=4 / 180s 这一点完整跑通**（exit 0，全部 artifacts 落盘）。
- 同一台机器上 c=1 / c=2 / c=8 的多次尝试，profiling 与请求都正常（mock 侧 100% 成功），但
  AIPerf 在 **Windows 上的结果总线偶发收不到 record**：日志停在
  `currently 0 of N records processed`，`profile_export.jsonl` 0 字节。这是 AIPerf 的 ZMQ 记录通道
  问题，不是 mock 或语料问题（同一次运行 `/metrics`、实时指标都正常）。已通过
  `AIPERF_SERVICE_WINDOWS_TCP_*` 缓解，但仍非 100% 稳定；Linux 上没有该问题。
- 因此当前只交付**一个完整并发点**；并发曲线用本仓库 `bench`/`agentx replay` 的合成会话补
  （见 `results/README.md`），两者不要混为一谈。

## 7. 与官网口径的差异（必须标注）

| 项 | 官网 | 本次 |
|---|---|---|
| 语料 | 393 条全量 | 8 条（真实前 8 条；全量重建需 ~30GB+ 内存） |
| 时长 | ≥900s，默认 3600s | 180s（`--unsafe-override`） |
| 模型 | 真实权重 + 厂商 chat template | mock（无语义） |
| 硬件指标 | throughput/chip、GPU 遥测、KV offload | 无（mock 无 GPU） |
| speculative acceptance | SPEED-Bench 固定值 | 不涉及 |

**结论**：这条链路证明「AgentX 官方 harness + 真实语料可以在离线环境跑起来并产出官方格式数据」，
可作为私有集群/网关/推理栈的**离线回归基线**；不能当作官方 leaderboard 分数。
