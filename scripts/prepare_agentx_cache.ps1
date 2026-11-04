# Prepare the offline cache baked into the AgentX image (Windows).
# Copies ONLY what the harness needs to run air-gapped:
#   - the Arrow cache of semianalysisai/cc-traces-weka-062126  (~1.8 GB)
#   - the DeepSeek-V3 tokenizer HF cache                        (~8 MB)
# Source: data/hf (populated by experiments/agentx_real/prefetch_dataset.py).
param(
    [string]$Source = "",
    [string]$Dest = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Source) { $Source = Join-Path $Root "data\hf" }
if (-not $Dest)   { $Dest   = Join-Path $Root "agentx_offline\cache" }

$dataset = Join-Path $Source "datasets\semianalysisai___cc-traces-weka-062126"
$tokenizer = Join-Path $Source "hub\models--deepseek-ai--DeepSeek-V3"
if (-not (Test-Path $dataset)) {
    throw "dataset cache missing: $dataset`nRun experiments/agentx_real/prefetch_dataset.py first."
}
if (-not (Test-Path $tokenizer)) {
    throw "tokenizer cache missing: $tokenizer`nLoad deepseek-ai/DeepSeek-V3 once (see docs/AGENTX_REAL.md)."
}

if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
New-Item -ItemType Directory -Force (Join-Path $Dest "datasets") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Dest "hub") | Out-Null

Write-Host "[cache] copying dataset arrow cache ..."
Copy-Item -Recurse -Force $dataset (Join-Path $Dest "datasets")
Write-Host "[cache] copying tokenizer ..."
Copy-Item -Recurse -Force $tokenizer (Join-Path $Dest "hub")

$size = (Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("[cache] ready: {0} ({1:N1} MB)" -f $Dest, $size)
Write-Host "[cache] next: docker build -t agentx-offline:1.0.7 agentx_offline"
