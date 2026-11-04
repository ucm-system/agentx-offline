# AgentX 离线测试镜像（agentx-offline）

一个**自带 AgentX 官方 harness + 真实语料**的离线测试镜像：在无外网的机房/实验室里，对任意
OpenAI 兼容的 LLM 推理服务跑 AgentX，并产出官方 artifacts + 可交互的 HTML 对比报告。

> 与 `gptzero:v3`（mock 服务镜像）的区别：
> `gptzero` 是**被测服务**（没有真模型，用于自测链路）；`agentx-offline` 是**测试工具**，
> 打的是你真实的 vLLM / SGLang / MindIE / TRT-LLM / 云 API 服务。

## 1. 镜像里有什么

| 组件 | 说明 |
|---|---|
| `agentx-harness==1.0.7` | 官方 harness（`aiperf` + AgentX scenario 锁） |
| 依赖 | 固定版本：`sentencepiece==0.2.0`（0.2.2 在部分机器上 DLL 崩溃）、transformers/datasets/pyarrow 等 |
| 数据集 | `semianalysisai/cc-traces-weka-062126` 的 Arrow 缓存（393 条 trace，1.77 GB） |
| Tokenizer | `deepseek-ai/DeepSeek-V3` 的 HF 缓存（8 MB） |
| 驱动 | `runner.py`：一个 target × 多个并发点，逐点跑 `agentx profile`，自动校验产物 |
| 报告 | `report.py`：多目标对比的**单文件 HTML**，ECharts 内联（无 CDN、无外网） |

镜像默认 `HF_HOME=/opt/hf` 且 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 / HF_DATASETS_OFFLINE=1`，
运行期**不需要任何外网**；只需要容器能访问被测服务的地址。

## 2. 构建与离线交付

已构建并推送的镜像（可直接在 ARM 服务器上拉取）：

```bash
docker pull ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64
```

自建构建：
```bash
# 0) 准备缓存（一次性；本仓库 data/hf 已由 experiments/agentx_real/prefetch_dataset.py 生成）
pwsh scripts/prepare_agentx_cache.ps1     # Windows
# bash scripts/prepare_agentx_cache.sh    # Linux/macOS

# 1) 构建（上下文 = agentx_offline/，约 3.6 GB）
docker build -t agentx-offline:1.0.7 agentx_offline

# 2) 导出成离线包
docker save -o dist/agentx-offline-1.0.7.tar agentx-offline:1.0.7
sha256sum dist/agentx-offline-1.0.7.tar > dist/agentx-offline-1.0.7.tar.sha256

# 3) 目标机器（无外网）
docker load -i agentx-offline-1.0.7.tar
docker run --rm agentx-offline:1.0.7 smoke     # 自检：数据集体+tokenizer 能否离线加载
```

精简版（不烤数据集，运行时挂卷）：

```bash
docker build --build-arg WITH_CACHE=0 -t agentx-offline:slim agentx_offline
docker run --rm -v /srv/hf:/opt/hf -v "$PWD/results:/results" agentx-offline:slim run ...
```

## 3. 对一个真实 LLM 服务跑

`--tokenizer` 接受三种写法：HuggingFace id（默认，镜像已烤好 DeepSeek-V3）、**本地模型或
tokenizer 目录**、或 `builtin`（零网络，按 `--model` 名生成）。模型目录在 `/mnt/model` 这类
位置时可以直接 `--tokenizer /mnt/model/Qwen3-8B`（软链接也可，只要容器内能解析）。

注意：官方语料的 hash block 是按 DeepSeek-V3 tokenizer 切分的，换 tokenizer 会改变 token
计数与 prefix-cache 命中率；要复现官方口径保持默认，做私有流量回归可以换模型自带 tokenizer。

```bash
docker run --rm \
  -v "$PWD/results:/results" \
  --add-host=host.docker.internal:host-gateway \
  agentx-offline:1.0.7 run \
    --url http://10.0.0.5:8000 \
    --model Qwen3-8B \
    --tokenizer deepseek-ai/DeepSeek-V3 \
    --target qwen3-8b-vllm \
    --concurrency 4,8,16 \
    --duration 900 \
    --entries 8
```

- `--url`：OpenAI 兼容 base URL（也接受完整的 `/v1/chat/completions`）；
- `--model`：服务端真实看到的 model id（多模型服务可分别跑）；
- `--duration`：每个并发点的 profiling 秒数；官方要求 ≥900，低于 900 会打 `submission_valid=false`；
- `--entries`：使用多少条真实 trace；393 条全量需要 ~30GB+ 内存，小机器用 8~32；
- `--concurrency`：并发 **agent 会话**数（不是 HTTP 批次），可逗号分隔多个点。

镜像内也会自动处理 Windows 风格端口窗口（`AIPERF_SERVICE_WINDOWS_TCP_*`，Linux 无影响）。

## 4. 多模型 / 多 infra 对比

同一个 results 卷里，每个 target 一个子目录；重复执行 `run` 时用不同 `--target`：

```bash
docker run --rm -v "$PWD/results:/results" agentx-offline:1.0.7 run \
  --url http://10.0.0.5:8000 --model Qwen3-8B --target qwen3-vllm \
  --concurrency 4,8,16 --duration 900

docker run --rm -v "$PWD/results:/results" agentx-offline:1.0.7 run \
  --url http://10.0.0.6:8000 --model Qwen3-8B --target qwen3-sglang \
  --concurrency 4,8,16 --duration 900

docker run --rm -v "$PWD/results:/results" agentx-offline:1.0.7 report --root /results
```

`report.html` 会给出：

- 每个 target 的 KPI 卡（峰值输出吞吐、最好 TTFT p50、服务端缓存命中）；
- 汇总表（请求、成功率、RPS、TTFT p50/p90/p99、ITL、E2E、tok/s、ISL/OSL、缓存理论/实测）；
- 6 张交互式图（TTFT p50、TTFT p90、输出吞吐、请求吞吐、E2E p90、缓存命中）按 target 对比；
- **曲线浏览器**（InferenceX 风格）：X / Y 各一个下拉框，任意指标组合（吞吐、交互性、TTFT p50/p90/p99、
  ITL、E2E、RPS、缓存、ISL/OSL、并发），可切对数轴；X 选「并发」时按顺序连线，选其他指标时
  自动变散点（吞吐-延迟前沿）；
- 每个 target 的时序页（profiling 每秒的 TTFT 与吞吐）。

图表为 ECharts 内联渲染，**单文件、可离线打开**（发给同事也不需要联网）。

## 5. 产物目录（完整数据）

```
results/
  report.html                       # 单文件交互报告
  report_data.json                  # 报告数据（机器可读）
  <target>/c<并发>/
    profile_export_aiperf.json      # 官方聚合指标
    profile_export_aiperf.csv
    profile_export.jsonl            # 逐请求记录
    profile_export_aiperf_timeslices.csv/.json
    server_metrics_export.json/.csv # 被测服务的 /metrics 快照（若暴露）
    profile_export_console.txt      # 官方 console 报告
    logs/aiperf.log                 # 全量日志（scenario 注入、每 30s 指标）
    run.log                         # 容器内驱动日志
  <target>/run_summary.json         # 该 target 的点位、耗时、退出码
```

## 6. 已验证（本机实测，2026-10-28）

| 项 | 结果 |
|---|---|
| 镜像 | `agentx-offline:1.0.7`，**3.92 GB**（Python 3.11 + agentx 1.0.7 + 1.77GB Arrow 语料 + tokenizer） |
| 离线自检 | `docker run --rm agentx-offline:1.0.7 smoke` → `dataset rows: 393`、`tokenizer ok`、`offline smoke OK` |
| 真跑 | 2 个 target（fast / slow）× c=4、c=8 × 120s，全部 `ok=True exit=0` |
| 报告 | 单文件 `report.html` 1.05 MB，离线可开，无外部请求 |

两个 target 是同一台机器上的 gptzero mock（模拟 fast/slow 两档 infra），用于验证链路：

| target | 并发 | 请求 | TTFT p50/p90/p99 (ms) | 交互性 tok/s | 输出 tok/s | 缓存命中（理论/实测） |
|---|---|---|---|---|---|---|
| mock-fast | 1 | 10 | 235 / 292 / 317 | 159 | 69 | 99.6% / 99.4% |
| mock-fast | 4 | 35 | 143 / 237 / 45,210* | 158 | 78 | 93.2% / 92.9% |
| mock-fast | 8 | 14 | 255 / 405 / 435 | 164 | 155 | 81.7% / 81.6% |
| mock-fast | 16 | 25 | 244 / 483 / 632 | 154 | 293 | 72.0% / 71.9% |
| mock-slow | 1 | 8 | 527 / 665 / 671 | 69 | 39 | 99.6% / 99.4% |
| mock-slow | 4 | 43 | 375 / 582 / 651 | 69 | 56 | 93.5% / 93.3% |
| mock-slow | 8 | 12 | 739 / 1,107 / 1,250 | 71 | 139 | 79.4% / 80.6% |
| mock-slow | 16 | 23 | 678 / 1,424 / 2,811 | 67 | 222 | 69.2% / 69.2% |

\* 真实离群点：一条超长上下文 prefill；原始记录见 `results/agentx_offline_demo/mock-fast/c4/profile_export.jsonl`。
报告里的 **逐请求延迟 CDF** 直接从这些 JSONL 记录（仅 profiling 阶段）生成，不依赖聚合摘要。

另外验证了**本地 trace 模式**：`--dataset-path /traces` 指向自建 WEKA 目录，同样产出完整
artifacts 与报告（该模式会打 `submission_valid=false`）。

完整 demo（报告 + 7 个点的全部官方 artifacts）：`results/agentx_offline_demo/`；
截图：`results/agentx_offline_demo/report_preview.png`。换成真实服务时只需改
`--url/--model/--target`，其余不变。

### 6.1 ARM64（Ascend 服务器实测）

镜像 `ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64`（arm64，3.57 GB）已推到
腾讯云 CCR，并在 一台 aarch64 / openEuler / 2 TB 内存 / 8×910B3 服务器上拉取验证：

| 步骤 | 结果 |
|---|---|
| `docker pull` | 成功（digest `sha256:ba8df24…`） |
| `smoke`（断网） | `dataset rows: 393`、`tokenizer ok`、`offline smoke OK` |
| `run` vs mock-fast c=4 | `ok=True exit=0`，54 请求，TTFT 184/252 ms，92.9 tok/s，缓存 94.9% |
| `run` vs mock-slow c=4 | `ok=True exit=0`，43 请求，TTFT 434/575 ms，55.6 tok/s，缓存 93.3% |

报告与数据：`results/arm_image/`（镜像内生成后拉回）；另有不用镜像、在 ARM 上原生跑
harness 的对照：`results/arm_server/`。ARM 镜像的构建方式见 `agentx_offline/Dockerfile`
（多阶段 wheelhouse，`crick` 在 builder 里用 cargo 编译）。

### 6.2 真实模型：Qwen3.5-0.8B @ 910B3（chip 7）

同一台机器上只占 **一张卡**，用 `vllm-ascend:v0.23.0` 起两个实例（顺序启动，避免共享卡 OOM）：

| target | 配置差异 | 并发 | TTFT p50/p90 | ITL p50 | 输出 tok/s |
|---|---|---|---|---|---|
| qwen35-base | 无 MTP、无 cudagraph、无 prefix cache、mem 0.30 | 4 / 8 | 3.8/5.6s · 1.2/5.0s | 22.4 / 22.1 ms | 82.9 / 152.3 |
| qwen35-mtp | MTP=1、cudagraph [2,16]、prefix cache、mem 0.40 | 4 / 8 | 1.7/8.5s · 4.8/12.9s | 15.3 / 63.6 ms | 81.6 / 80.3 |

要点：

- 服务必须 `--dtype bfloat16`；`float16` 下 GDN 算子 `aclnnChunkGatedDeltaRuleFwdH` 直接报
  dtype 不匹配，起不来；
- 官方 393 条语料峰值上下文全部 >32k，`max-model-len=16384` 装不下，因此用本地 13k 合成
  trace + `--max-context 16000`，该模式会打 `submission_valid=false`；
- `--tokenizer` 用本地目录 `/data/model/Qwen3.5-0.8B`：镜像已 patch aiperf 1.0.7 的
  `_resolve_local_snapshot()`（上游把本地目录当 HF repo id，离线必挂）；
- MTP 在 c=4 把 ITL 从 22.4→15.3ms，但 c=8 反而 63.6ms（cudagraph 捕获尺寸/高并发回退），
  单卡共享集群上的 120s 结果只用于验证链路和相对趋势。

完整说明与产物：`results/real_qwen35/`。



## 7. 已知边界

- 镜像里的语料是 Arrow 缓存，**不含**原始 1.85GB `traces.jsonl`（离线复现只需要 Arrow）；
- 393 条全量需要大内存机器（重建会并行展开会话），`--entries` 是离线小机器的必要开关；
- 官方 submission 还要求 pinned 语料 + ≥900s + 真实模型服务；本镜像满足前三者的形式要求，
  但分数的解释权仍取决于你的服务是否真实（mock 分数没有意义）；
- GPU 遥测默认关闭（`--no-gpu-telemetry`）；需要的话在 `--extra` 里打开并挂载 GPU 设备。

## 8. 怎么读这份报告（与 AgentX 官方口径对照）

| 报告里的列/图 | 官方口径 | 注意 |
|---|---|---|
| TTFT p50/p90/p99 | AgentX 的核心交互性指标 | 142k 上下文下 p99 长尾很正常；不要只看 p50 |
| ITL p50 / 交互性 tok/s | 官方 `intvty = 1/TPOT`，衡量“人可感知的生成速度” | 与总吞吐是两回事：低并发看 ITL，高并发看吞吐 |
| 输出 tok/s | 官方按 `throughput/chip` 归一化到芯片数 | 本镜像不知道你的卡数，报告给**整机**吞吐；需要 per-chip 自己除 |
| 请求吞吐 RPS | 受 trace 时序约束（闭环） | 并发提高不一定线性涨，AgentX 不是固定批次压测 |
| E2E p50/p90 | 请求全程 | 受长 prefill + 排队共同影响 |
| 缓存命中（理论/实测） | 官方明确 98% cached input 是 AgentX 负载特征 | 理论值来自 trace 的 LCP；实测低说明服务端 KV 容量/前缀缓存策略有差距 |
| 汇总表里的 ISL/OSL | 官方 median in=142k / out=444 | 用 `--entries 8` 时是子集，中位数会偏小，属预期 |

**关键提醒**：AgentX 是**闭环** benchmark，`--concurrency` 是并发 agent 会话数，不是 HTTP 批次；
不同并发点完成的请求数不同（快服务跑得更多），所以吞吐和延迟要成对看，不能只比单点。

## 9. 更多测试：本地 trace 模式

除了官方 pinned 语料，也可以直接喂本地 WEKA trace 目录（例如自建采集）：

```bash
docker run --rm -v /data/my-traces:/traces -v "$PWD/results:/results" agentx-offline:1.0.7 run \
  --url http://llm:8000 --model Qwen3-8B --target my-traces \
  --dataset-path /traces --concurrency 4 --duration 300
```

此时 harness 会打 `submission_valid=false`（本地语料身份不可核验），但 artifacts 与报告流程完全一致，
适合做**私有流量回归**。该模式已实测跑通（镜像内 `--input-file /traces`）。

## 10. 语料子集代表性（互动调研数据）

镜像里虽然烤了 393 条全量，但小机器跑 `--entries 8`。实测这 8 条与全量语料的分布对比：

| 指标 | 前 8 条子集 | 全量 393 条 | 官方公布 |
|---|---|---|---|
| 请求数 | 1,892 | 98,827 | 98,827 |
| 输入 token 中位数 | 133,792 | 142,016 | 142,016 |
| 输出 token 中位数 | 412 | 444 | 444 |

子集的请求级中位数与全量非常接近（长上下文、短输出的 agentic 形状没变）；报告里出现更低的
ISL p50（40k~70k）是因为 AgentX 在每条 trace 的 25%~75% 处取轨迹快照，profiling 窗口内只覆盖
部分上下文，不是语料被截断。

## 11. GitHub Actions 构建（x86 + ARM 原生）

工作流：`.github/workflows/agentx-image.yml`。利用 GitHub 自带的两种 runner 原生构建，避免 QEMU：

| 矩阵 | runner | 产物 |
|---|---|---|
| amd64 | `ubuntu-24.04` | `ghcr.io/<owner>/agentx-offline:<version>-amd64[-slim]` |
| arm64 | `ubuntu-24.04-arm` | `ghcr.io/<owner>/agentx-offline:<version>-arm64[-slim]` |

用法：Actions → **agentx-offline image** → Run workflow，填 `version`（如 `1.0.7`）、
`with_cache`（是否烤 393 条语料）、`push`。也可以 push `agentx-v1.0.7` 这类 tag 触发
（默认构建 slim，不烤语料）。

- **slim**（默认）：不烤语料，镜像约 2 GB，运行时 `-v /srv/hf:/opt/hf` 挂缓存；
- **with_cache**：每个 arch job 先构建 slim、用它跑 `prefetch_cache.py` 把语料 + tokenizer
  下载到构建上下文，再构建完整镜像（约 3.9 GB）；两条 arch 都成功后自动合成
  `:<version>` 多架构 manifest；
- 想推腾讯云 CCR 而不是 GHCR，把 `docker/login-action` 的 registry 换成
  `ccr.ccs.tencentyun.com`、`IMAGE_NAME` 换成 `keriko/agentx-offline`，并在仓库 secrets 里放
  `CCR_USERNAME` / `CCR_PASSWORD`；
- `crick` 没有 linux-aarch64 wheel，ARM job 会在 builder 阶段用 apt 装的 cargo 编译；
  构建缓存用 `type=gha`，第二次起会快很多。

> 注意：`agentx_offline/cache/` 被 `.gitignore` 排除，CI 里的空目录由 workflow 创建；
> 不要把本地 1.8 GB 语料提交进仓库。

## 12. 256k 上下文的真实模型（官方语料）

Qwen3.5-0.8B 的 `max_position_embeddings=262144`，独占 chip 7、`--gpu-memory-utilization 0.85`
可以把 `--max-model-len` 拉到 262144，从而直接跑**官方 pinned 语料**（不再需要本地 trace）：

```bash
docker run --rm --net host -v /data/agentx/real_results:/results \
  -v /data/model:/data/model:ro \
  ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64 run \
    --url http://127.0.0.1:8080 --model qwen3.5 \
    --tokenizer /data/model/Qwen3.5-0.8B \
    --target qwen35-256k --concurrency 4 --duration 300 \
    --entries 64 --max-context 250000
```

实测（c=4、300s、64 条官方 trace、27 个请求完成）：TTFT p50/p90 = 7.0/9.8 s，
ITL p50 = 9.5 ms，输出 35 tok/s；c=8 时 TTFT 10.2/17.6 s、ITL 44.9 ms、输出 56 tok/s。
ISL p50（44k / 96k）是轨迹切片后的窗口值，不是语料被裁。
`--max-context 250000` 只是过滤峰值超过模型窗口的 trace，其余仍是官方 pinned 数据。

## 13. 按上下文窗口跑（64K / 128K / 256K）与 SLA 自动调优

AgentX 用 `--max-context-length` 过滤「峰值 prompt+output」超过窗口的 trace；本镜像把常用窗口
做成标签，`--window 64k` 等价于 `--max-context-length 65024`（留 512 token 余量）：

| 窗口 | 可用官方 trace（393 条中） | 命令 |
|---|---|---|
| 32K | 0（最小一条峰值 ~43k） | 不可行 |
| 64K | 9 | `--window 64k --entries 9` |
| 128K | 85 | `--window 128k --entries 32`（按机器内存取前 N 条） |
| 256K | 221 | `--window 256k --entries 64` |
| 512K | 约 300+ | `--window 512k` |
| 1M | 393（全量） | 不加过滤，需要 ≥1M 窗口的模型 |

注意：窗口越小，可用 trace 越少，且这些 trace 的 ISL 分布会被窗口“截断”成偏短样本；跨窗口比较时
要同时看 `entries` 和 ISL，不能只比吞吐。

### SLA 自动调优

`tune` 子命令会自动扫并发、按 SLA 过滤、选吞吐最大的点，并写 `tune.json` / `tune.md` / `report.html`：

```bash
docker run --rm --net host -v /data/agentx/results_windows:/results \
  -v /mnt/model:/mnt/model:ro -v /data/model:/data/model:ro \
  ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64 tune \
    --url http://127.0.0.1:8080 --model qwen3.5 \
    --tokenizer /data/model/Qwen3.5-0.8B \
    --target qwen35-64k --window 64k \
    --concurrency 2,4,8,16 \
    --duration 120 --entries 9 \
    --sla-ttft-p90 5000 --sla-itl-p50 50 --sla-min-success 99
```

或用自适应阶梯（每轮翻倍，连续 2 个点不满足 SLA / 吞吐下滑就停）：

```bash
... tune --auto --start 1 --max-concurrency 128 \
    --window 64k --sla-ttft-p90 5000 --sla-itl-p50 50
```

判据：每个点读取官方 `profile_export_aiperf.json` 的 TTFT p90 / ITL p50 / E2E p90 / 成功率，
全部满足才算 PASS；推荐值是 **PASS 集合里 out_tps 最大的并发**。没有 PASS 点时退回全局最大吞吐并标注。



