#!/usr/bin/env python3
"""Build a self-contained HTML report from AgentX run artifacts.

Reads the official ``agentx`` output (``profile_export_aiperf.json``,
``profile_export_console.txt``, ``profile_export_aiperf_timeslices.csv``,
``server_metrics_export.json``) and renders:

* a multi-target comparison table (model / infra A vs B),
* interactive ECharts (TTFT, throughput, cache reuse, E2E) — no CDN, the JS is
  inlined into the HTML so the file works in an air-gapped environment,
* a per-target timeline tab (TTFT and throughput over the profiling window).

No matplotlib, no external assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ECHARTS_FILE = ASSET_DIR / "echarts.min.js"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _stat(block: dict | None, key: str) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    return round(float(value), 3) if value is not None else None


def _last_group(text: str, pattern: str) -> float | None:
    matches = re.findall(pattern, text)
    return round(float(matches[-1]), 2) if matches else None


def _timeslices(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    if not path.exists():
        return out
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    wanted = {
        ("Time to First Token", "p50"): "ttft_p50",
        ("Time to First Token", "p90"): "ttft_p90",
        ("Output Token Throughput", "avg"): "out_tps",
        ("Request Throughput", "avg"): "req_tps",
        ("Inter Token Latency", "p50"): "itl_p50",
    }
    for row in rows:
        key = wanted.get((row["Metric"], row["Stat"]))
        if key:
            out.setdefault(key, []).append(round(float(row["Value"]), 3))
    return out


def _cdf(values: list[float], max_points: int = 240) -> list[list[float]]:
    """Empirical CDF points ``[[latency_ms, percentile], ...]``."""
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return []
    n = len(clean)
    step = max(1, n // max_points)
    points: list[list[float]] = []
    for i in range(0, n, step):
        points.append([round(clean[i], 1), round(100.0 * (i + 1) / n, 2)])
    if points[-1][1] < 100.0:
        points.append([round(clean[-1], 1), 100.0])
    return points


def _records_cdf(point_dir: Path) -> dict[str, list[list[float]]]:
    """Read the official per-request JSONL and keep profiling-phase latencies only."""
    path = point_dir / "profile_export.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    ttft: list[float] = []
    e2e: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = rec.get("metadata") or {}
            if meta.get("benchmark_phase") != "profiling":
                continue
            metrics = rec.get("metrics") or {}
            for key, bucket in (("time_to_first_token", ttft), ("request_latency", e2e)):
                value = (metrics.get(key) or {}).get("value")
                if value is not None:
                    bucket.append(float(value))
    return {"ttft": _cdf(ttft), "e2e": _cdf(e2e)}


def load_point(point_dir: Path, target: str, concurrency: int) -> dict | None:
    summary = point_dir / "profile_export_aiperf.json"
    if not summary.exists() or summary.stat().st_size == 0:
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    console = ""
    console_file = point_dir / "profile_export_console.txt"
    if console_file.exists():
        console = console_file.read_text(encoding="utf-8", errors="replace")
    log_file = point_dir / "logs" / "aiperf.log"
    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""

    requests = int((data.get("request_count") or {}).get("avg") or 0)
    errors = len(data.get("error_summary") or [])
    point = {
        "target": target,
        "concurrency": concurrency,
        "path": str(point_dir),
        "requests": requests,
        "errors": errors,
        "success_pct": round(100.0 * (requests - errors) / requests, 1) if requests else None,
        "rps": _stat(data.get("request_throughput"), "avg"),
        "ttft": {
            k: _stat(data.get("time_to_first_token"), k) for k in ("p50", "p90", "p99", "avg")
        },
        "itl": {k: _stat(data.get("inter_token_latency"), k) for k in ("p50", "p90")},
        "e2e": {k: _stat(data.get("request_latency"), k) for k in ("p50", "p90", "p99")},
        "out_tps": _stat(data.get("output_token_throughput"), "avg"),
        "in_tps": _stat(data.get("input_token_throughput"), "avg"),
        "isl_p50": _stat(data.get("input_sequence_length"), "p50"),
        "osl_p50": _stat(data.get("output_sequence_length"), "p50"),
        "cache_theory_pct": _last_group(
            console, r"Theoretical Prefix Cache Hit \(%\)\s*│\s*([\d.]+)"
        ),
        "cache_server_pct": _last_group(log_text, r"prefix_cache_hit=([\d.]+)%"),
        "interactivity": round(1000.0 / data["inter_token_latency"]["p50"], 1)
        if (data.get("inter_token_latency") or {}).get("p50")
        else None,
        "timeline": _timeslices(point_dir / "profile_export_aiperf_timeslices.csv"),
        "cdf": _records_cdf(point_dir),
        "artifacts": {
            "summary": str(summary.relative_to(point_dir.parent.parent.parent)),
            "records": f"{target}/c{concurrency}/profile_export.jsonl",
        },
    }
    return point


def discover(root: Path) -> tuple[list[dict], list[str]]:
    points: list[dict] = []
    targets: list[str] = []
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for conc_dir in sorted(target_dir.glob("c*")):
            m = re.fullmatch(r"c(\d+)", conc_dir.name)
            if not m:
                continue
            point = load_point(conc_dir, target_dir.name, int(m.group(1)))
            if point:
                points.append(point)
                if target_dir.name not in targets:
                    targets.append(target_dir.name)
    points.sort(key=lambda p: (p["target"], p["concurrency"]))
    return points, targets


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "–"
    return f"{value:,.{digits}f}{suffix}"


def _table(points: list[dict]) -> str:
    head = (
        "<tr><th>目标 / 模型</th><th>并发</th><th>请求</th><th>成功</th><th>RPS</th>"
        "<th>TTFT p50/p90/p99 (ms)</th><th>ITL p50 (ms)</th><th>交互性 tok/s</th><th>E2E p50/p90 (ms)</th>"
        "<th>输出 tok/s</th><th>输入 tok/s</th><th>ISL p50</th><th>OSL p50</th>"
        "<th>缓存命中 理论/实测</th></tr>"
    )
    rows = []
    for p in points:
        rows.append(
            "<tr>"
            f"<td class='target'>{p['target']}</td>"
            f"<td class='num'>{p['concurrency']}</td>"
            f"<td class='num'>{p['requests']}</td>"
            f"<td class='num'>{_fmt(p['success_pct'], 1, '%')}</td>"
            f"<td class='num'>{_fmt(p['rps'], 2)}</td>"
            f"<td class='num'>{_fmt(p['ttft']['p50'], 0)} / {_fmt(p['ttft']['p90'], 0)} / {_fmt(p['ttft']['p99'], 0)}</td>"
            f"<td class='num'>{_fmt(p['itl']['p50'], 1)}</td>"
            f"<td class='num'>{_fmt(p['interactivity'], 0)}</td>"
            f"<td class='num'>{_fmt(p['e2e']['p50'], 0)} / {_fmt(p['e2e']['p90'], 0)}</td>"
            f"<td class='num'>{_fmt(p['out_tps'], 0)}</td>"
            f"<td class='num'>{_fmt(p['in_tps'], 0)}</td>"
            f"<td class='num'>{_fmt(p['isl_p50'], 0)}</td>"
            f"<td class='num'>{_fmt(p['osl_p50'], 0)}</td>"
            f"<td class='num'>{_fmt(p['cache_theory_pct'], 1, '%')} / {_fmt(p['cache_server_pct'], 1, '%')}</td>"
            "</tr>"
        )
    return f"<table class='grid'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _kpi_cards(points: list[dict], targets: list[str]) -> str:
    cards = []
    for target in targets:
        pts = [p for p in points if p["target"] == target]
        peak = max(pts, key=lambda p: p["out_tps"] or 0)
        best_ttft = min(pts, key=lambda p: p["ttft"]["p50"] or 1e9)
        cache = [p for p in pts if p["cache_server_pct"] is not None]
        cache_pct = max(cache, key=lambda p: p["concurrency"])["cache_server_pct"] if cache else None
        cards.append(
            f"""
            <div class="card">
              <div class="card-title">{target}</div>
              <div class="kpis">
                <div><span class="kpi">{_fmt(peak['out_tps'], 0)}</span><span class="unit">out tok/s</span>
                     <div class="sub">@ c={peak['concurrency']}</div></div>
                <div><span class="kpi">{_fmt(best_ttft['ttft']['p50'], 0)}</span><span class="unit">ms TTFT p50</span>
                     <div class="sub">@ c={best_ttft['concurrency']}</div></div>
                <div><span class="kpi">{_fmt(cache_pct, 1, '%')}</span><span class="unit">cache hit</span>
                     <div class="sub">服务端实测</div></div>
              </div>
            </div>"""
        )
    return "".join(cards)


def _options(points: list[dict], targets: list[str]) -> dict:
    concs = sorted({p["concurrency"] for p in points})
    colors = ["#4f8cff", "#22c3a6", "#f5a524", "#ef5da8", "#9b7bff"]

    def line_series(metric_path: str, name: str) -> list[dict]:
        series = []
        for i, target in enumerate(targets):
            data = []
            for c in concs:
                match = next((p for p in points if p["target"] == target and p["concurrency"] == c), None)
                value = None
                if match:
                    node = match
                    for key in metric_path.split("."):
                        node = node.get(key) if isinstance(node, dict) else None
                    value = node
                data.append(value)
            series.append(
                {
                    "name": target,
                    "type": "line",
                    "smooth": True,
                    "symbolSize": 7,
                    "data": data,
                    "itemStyle": {"color": colors[i % len(colors)]},
                }
            )
        return series

    base = {
        "backgroundColor": "transparent",
        "grid": {"left": 56, "right": 28, "top": 42, "bottom": 48},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "textStyle": {"color": "#9fb0c6"}},
        "xAxis": {
            "type": "category",
            "data": [str(c) for c in concs],
            "axisLine": {"lineStyle": {"color": "#324156"}},
            "axisLabel": {"color": "#9fb0c6"},
        },
        "yAxis": {
            "type": "value",
            "axisLine": {"show": False},
            "splitLine": {"lineStyle": {"color": "#1d2836"}},
            "axisLabel": {"color": "#9fb0c6"},
        },
    }

    def axis(name: str) -> dict:
        opt = json.loads(json.dumps(base))
        opt["yAxis"]["name"] = name
        return opt

    ttft50 = axis("ms")
    ttft50["title"] = {"text": "TTFT p50", "left": "center", "top": 6, "textStyle": {"color": "#dbe6f3", "fontSize": 13}}
    ttft50["series"] = line_series("ttft.p50", "TTFT p50")

    ttft90 = axis("ms")
    ttft90["title"] = {"text": "TTFT p90", "left": "center", "top": 6, "textStyle": {"color": "#dbe6f3", "fontSize": 13}}
    ttft90["series"] = line_series("ttft.p90", "TTFT p90")

    throughput = axis("tokens/s")
    throughput["title"] = {"text": "输出吞吐", "left": "center", "top": 6, "textStyle": {"color": "#dbe6f3", "fontSize": 13}}
    throughput["series"] = line_series("out_tps", "out tok/s")

    rps = axis("req/s")
    rps["title"] = {"text": "请求吞吐", "left": "center", "top": 6, "textStyle": {"color": "#dbe6f3", "fontSize": 13}}
    rps["series"] = line_series("rps", "req/s")

    e2e = axis("ms")
    e2e["title"] = {"text": "E2E p90", "left": "center", "top": 6, "textStyle": {"color": "#dbe6f3", "fontSize": 13}}
    e2e["series"] = line_series("e2e.p90", "E2E p90")

    cache = axis("%")
    cache["title"] = {"text": "前缀缓存命中（服务端实测）", "left": "center", "top": 6, "textStyle": {"color": "#dbe6f3", "fontSize": 13}}
    cache["series"] = line_series("cache_server_pct", "cache hit")
    cache["yAxis"]["max"] = 100

    return {
        "ttft50": ttft50,
        "ttft90": ttft90,
        "throughput": throughput,
        "rps": rps,
        "e2e": e2e,
        "cache": cache,
    }

def _cdf_options(points: list[dict], targets: list[str]) -> dict:
    """Latency CDF at the highest concurrency available per target."""
    colors = ["#4f8cff", "#22c3a6", "#f5a524", "#ef5da8", "#9b7bff"]

    def build(kind: str, title: str) -> dict:
        series = []
        for i, target in enumerate(targets):
            pts = [p for p in points if p["target"] == target and p.get("cdf", {}).get(kind)]
            if not pts:
                continue
            peak = max(pts, key=lambda p: p["concurrency"])
            series.append(
                {
                    "name": f"{target} · c{peak['concurrency']}",
                    "type": "line",
                    "showSymbol": False,
                    "step": "end",
                    "data": peak["cdf"][kind],
                    "itemStyle": {"color": colors[i % len(colors)]},
                }
            )
        return {
            "backgroundColor": "transparent",
            "grid": {"left": 52, "right": 28, "top": 42, "bottom": 52},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "textStyle": {"color": "#9fb0c6"}},
            "title": {"text": title, "left": "center", "top": 6,
                      "textStyle": {"color": "#dbe6f3", "fontSize": 13}},
            "xAxis": {
                "type": "value",
                "name": "ms",
                "axisLabel": {"color": "#9fb0c6"},
                "axisLine": {"lineStyle": {"color": "#324156"}},
                "splitLine": {"lineStyle": {"color": "#1d2836"}},
            },
            "yAxis": {
                "type": "value",
                "name": "%",
                "max": 100,
                "axisLabel": {"color": "#9fb0c6"},
                "splitLine": {"lineStyle": {"color": "#1d2836"}},
            },
            "series": series,
        }

    return {"ttft": build("ttft", "TTFT CDF（最高并发）"), "e2e": build("e2e", "E2E CDF（最高并发）")}


def _timeline_cap(values: list[float]) -> float | None:
    """Clip the timeline y-axis so one cold-start spike does not flatten the run.

    Uses 4x the median p50 (bounded to 500..5000 ms); values above are shown
    against the top edge by ECharts' clipping, and the raw data is untouched.
    """
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    median = clean[len(clean) // 2]
    return max(500.0, min(5000.0, median * 4.0))


_EXPLORER_METRICS = [
    ("concurrency", "并发", "", False),
    ("out_tps", "输出吞吐", "tok/s", True),
    ("in_tps", "输入吞吐", "tok/s", True),
    ("interactivity", "交互性 1/TPOT", "tok/s", True),
    ("ttft_p50", "TTFT p50", "ms", True),
    ("ttft_p90", "TTFT p90", "ms", True),
    ("ttft_p99", "TTFT p99", "ms", True),
    ("itl_p50", "ITL p50", "ms", True),
    ("e2e_p50", "E2E p50", "ms", True),
    ("e2e_p90", "E2E p90", "ms", True),
    ("rps", "请求吞吐", "req/s", True),
    ("cache_theory_pct", "缓存命中（理论）", "%", False),
    ("cache_server_pct", "缓存命中（服务端）", "%", False),
    ("isl_p50", "ISL p50", "tokens", True),
    ("osl_p50", "OSL p50", "tokens", True),
]


def _point_metrics(point: dict) -> dict:
    return {
        "concurrency": point["concurrency"],
        "out_tps": point.get("out_tps"),
        "in_tps": point.get("in_tps"),
        "interactivity": point.get("interactivity"),
        "ttft_p50": point["ttft"].get("p50"),
        "ttft_p90": point["ttft"].get("p90"),
        "ttft_p99": point["ttft"].get("p99"),
        "itl_p50": point["itl"].get("p50"),
        "e2e_p50": point["e2e"].get("p50"),
        "e2e_p90": point["e2e"].get("p90"),
        "rps": point.get("rps"),
        "cache_theory_pct": point.get("cache_theory_pct"),
        "cache_server_pct": point.get("cache_server_pct"),
        "isl_p50": point.get("isl_p50"),
        "osl_p50": point.get("osl_p50"),
    }


def _explorer_options(points: list[dict], targets: list[str]) -> dict:
    """Point-level payload for the switchable X/Y explorer chart."""
    metrics = [
        {"key": key, "label": label, "unit": unit, "log": log}
        for key, label, unit, log in _EXPLORER_METRICS
    ]
    series = []
    for target in targets:
        pts = sorted(
            (p for p in points if p["target"] == target), key=lambda p: p["concurrency"]
        )
        series.append({"name": target, "points": [_point_metrics(p) for p in pts]})
    return {
        "metrics": metrics,
        "series": series,
        "defaultX": "out_tps",
        "defaultY": "interactivity",
    }


EXPLORER_JS = r"""
const EXPLORER = __EXPLORER__;
(function () {
  const el = document.getElementById("c-explorer");
  if (!el) return;
  const chart = echarts.init(el, null, {renderer: "canvas"});
  const xs = document.getElementById("x-metric");
  const ys = document.getElementById("y-metric");
  const logEl = document.getElementById("log-axes");
  const metric = key => EXPLORER.metrics.find(m => m.key === key);
  const label = m => m.label + (m.unit ? " (" + m.unit + ")" : "");
  for (const m of EXPLORER.metrics) {
    xs.add(new Option(label(m), m.key));
    ys.add(new Option(label(m), m.key));
  }
  xs.value = EXPLORER.defaultX;
  ys.value = EXPLORER.defaultY;
  function build() {
    const xm = metric(xs.value), ym = metric(ys.value);
    const isLog = m => logEl.checked && m.log;
    const series = EXPLORER.series.map(s => ({
      name: s.name,
      // Lines only make sense when the X axis is the concurrency sweep;
      // for metric-vs-metric views draw a scatter (Pareto-style).
      type: xm.key === "concurrency" ? "line" : "scatter",
      symbolSize: xm.key === "concurrency" ? 8 : 12,
      connectNulls: true,
      emphasis: {focus: "series"},
      data: s.points
        .map(p => [p[xm.key], p[ym.key], p.concurrency])
        .filter(d => d[0] != null && d[1] != null)
        .filter(d => (!isLog(xm) || d[0] > 0) && (!isLog(ym) || d[1] > 0)),
    }));
    return {
      backgroundColor: "transparent",
      grid: {left: 76, right: 32, top: 36, bottom: 78},
      tooltip: {
        trigger: "item",
        formatter: params => {
          const d = params.data;
          return "<b>" + params.seriesName + "</b> · c" + d[2] +
            "<br>" + label(xm) + ": " + d[0] +
            "<br>" + label(ym) + ": " + d[1];
        },
      },
      legend: {bottom: 0, textStyle: {color: "#9fb0c6"}},
      xAxis: {
        type: isLog(xm) ? "log" : "value",
        name: label(xm),
        nameLocation: "middle",
        nameGap: 32,
        axisLabel: {color: "#9fb0c6"},
        splitLine: {lineStyle: {color: "#1d2836"}},
      },
      yAxis: {
        type: isLog(ym) ? "log" : "value",
        name: label(ym),
        nameLocation: "middle",
        nameGap: 52,
        axisLabel: {color: "#9fb0c6"},
        splitLine: {lineStyle: {color: "#1d2836"}},
      },
      series,
    };
  }
  function render() { chart.setOption(build(), true); }
  xs.addEventListener("change", render);
  ys.addEventListener("change", render);
  logEl.addEventListener("change", render);
  window.addEventListener("resize", () => chart.resize());
  render();
})();
"""


def _timeline_options(points: list[dict]) -> dict:
    out = {}
    for p in points:
        tl = p.get("timeline") or {}
        p50 = tl.get("ttft_p50") or []
        if not p50:
            continue
        key = f"{p['target']}·c{p['concurrency']}"
        seconds = list(range(len(p50)))
        series = [
            {
                "name": "TTFT p50",
                "type": "line",
                "showSymbol": False,
                "data": p50,
                "itemStyle": {"color": "#4f8cff"},
            }
        ]
        if tl.get("out_tps"):
            series.append(
                {
                    "name": "out tok/s",
                    "type": "line",
                    "showSymbol": False,
                    "yAxisIndex": 1,
                    "data": tl["out_tps"],
                    "itemStyle": {"color": "#22c3a6"},
                }
            )
        out[key] = {
            "backgroundColor": "transparent",
            "grid": {"left": 56, "right": 28, "top": 44, "bottom": 48},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "textStyle": {"color": "#9fb0c6"}},
            "title": {"text": key, "left": "center", "top": 6, "textStyle": {"color": "#dbe6f3", "fontSize": 13}},
            "xAxis": {
                "type": "category",
                "data": seconds,
                "axisLabel": {"color": "#9fb0c6"},
                "axisLine": {"lineStyle": {"color": "#324156"}},
            },
            "yAxis": [
                {
                    "type": "value",
                    "name": "ms",
                    "max": _timeline_cap(p50),
                    "splitLine": {"lineStyle": {"color": "#1d2836"}},
                    "axisLabel": {"color": "#9fb0c6"},
                },
                {
                    "type": "value",
                    "name": "tok/s",
                    "splitLine": {"show": False},
                    "axisLabel": {"color": "#9fb0c6"},
                },
            ],
            "series": series,
        }
    return out


CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b0f14;color:#dbe6f3;font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
header{padding:22px 28px 8px}
h1{font-size:20px;margin:0 0 6px;letter-spacing:.2px}
.meta{color:#8ea3bb;font-size:13px;line-height:1.8}
.wrap{padding:8px 28px 36px;max-width:1560px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:18px 0}
.card{background:linear-gradient(180deg,#121a24,#0f1620);border:1px solid #1e2a38;border-radius:12px;padding:14px 16px}
.card-title{font-size:13px;color:#9fb0c6;margin-bottom:10px;letter-spacing:.3px}
.kpis{display:flex;gap:18px;flex-wrap:wrap}
.kpi{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.unit{font-size:11px;color:#7f90a6;margin-left:5px}
.sub{font-size:11px;color:#63758c;margin-top:2px}
.panel{background:#101822;border:1px solid #1e2a38;border-radius:14px;padding:14px 16px;margin:14px 0}
.panel h2{font-size:14px;margin:0 0 8px;color:#c9d8ea;font-weight:600}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.chart{height:300px}
table.grid{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
table.grid th{color:#8ea3bb;text-align:left;font-weight:500;padding:9px 8px;border-bottom:1px solid #1e2a38;white-space:nowrap}
table.grid td{padding:8px;border-bottom:1px solid #16202c;white-space:nowrap}
table.grid td.num{text-align:right}
table.grid td.target{color:#9ec1ff;font-weight:600}
table.grid tr:hover td{background:#131d28}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 12px}
.tab{padding:6px 12px;border-radius:999px;border:1px solid #223142;color:#9fb0c6;font-size:12px;cursor:pointer;background:#0f1620}
.tab.active{background:#17263a;color:#dbe6f3;border-color:#2f4a68}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:2px 0 10px;color:#9fb0c6;font-size:13px}
.controls select{background:#0f1620;color:#dbe6f3;border:1px solid #223142;border-radius:8px;padding:6px 10px;font-size:13px}
.controls .hint{color:#6f8199;font-size:12px}
.note{color:#8ea3bb;font-size:13px;line-height:1.8}
footer{color:#7f90a6;font-size:12px;margin-top:24px}
a{color:#6ea8ff;text-decoration:none}
"""


def render_html(points: list[dict], targets: list[str], echarts_js: str) -> str:
    concs = sorted({p["concurrency"] for p in points})
    options = _options(points, targets)
    options["cdf"] = _cdf_options(points, targets)
    timelines = _timeline_options(points)
    explorer = _explorer_options(points, targets)
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    table = _table(points)
    cards = _kpi_cards(points, targets)

    timeline_tabs = "".join(
        f'<div class="tab{" active" if i == 0 else ""}" data-key="{key}">{key}</div>'
        for i, key in enumerate(timelines)
    )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentX 对比报告</title>
<style>{CSS}</style>
<script>{echarts_js}</script>
</head><body>
<header>
  <h1>AgentX 推理性能对比报告</h1>
  <div class="meta">
    目标：{", ".join(targets)} · 并发点：{", ".join(map(str, concs))} ·
    数据集：semianalysisai/cc-traces-weka-062126 · 生成于 {generated}
    <br>所有数字直接来自 agentx-harness 的 profile_export_aiperf.json / server_metrics_export.json；
    AgentX 是闭环 agent 回放，并发数是「并发 agent 会话」而不是固定请求批次。
  </div>
</header>
<div class="wrap">
  <div class="cards">{cards}</div>

  <div class="panel">
    <h2>汇总表</h2>
    {table}
  </div>

  <div class="panel">
    <h2>曲线浏览器（X / Y 可切换）</h2>
    <div class="controls">
      <label>X 轴 <select id="x-metric"></select></label>
      <label>Y 轴 <select id="y-metric"></select></label>
      <label><input type="checkbox" id="log-axes" checked> 对数轴</label>
      <span class="hint">每个点是一个并发点；X 选「并发」时按顺序连线，选其他指标时是散点（吞吐-延迟前沿），可任意组合 TTFT / ITL / E2E / 缓存。</span>
    </div>
    <div class="chart" id="c-explorer" style="height:460px"></div>
  </div>

  <div class="panel">
    <h2>横向对比</h2>
    <div class="charts">
      <div class="chart" id="c-ttft50"></div>
      <div class="chart" id="c-ttft90"></div>
      <div class="chart" id="c-throughput"></div>
      <div class="chart" id="c-rps"></div>
      <div class="chart" id="c-e2e"></div>
      <div class="chart" id="c-cache"></div>
    </div>
  </div>

  <div class="panel">
    <h2>逐请求延迟 CDF（profiling 窗口）</h2>
    <div class="charts">
      <div class="chart" id="c-cdf-ttft"></div>
      <div class="chart" id="c-cdf-e2e"></div>
    </div>
  </div>

  <div class="panel">
    <h2>时序（profiling 窗口内每秒）</h2>
    <div class="tabs">{timeline_tabs}</div>
    <div class="chart" id="c-timeline" style="height:340px"></div>
  </div>

  <div class="panel">
    <h2>原始数据</h2>
    <div class="note">
      每个并发点的完整 agentx 产物都在结果目录：<code>&lt;target&gt;/c&lt;并发&gt;/</code>，
      含 <code>profile_export_aiperf.json</code>（聚合）、<code>profile_export.jsonl</code>（逐请求）、
      <code>profile_export_aiperf_timeslices.csv</code>（1s 切片）、<code>server_metrics_export.json</code>、
      <code>profile_export_console.txt</code> 与 <code>logs/aiperf.log</code>。
    </div>
  </div>
  <footer>agentx-offline · 纯离线渲染（ECharts 内联，无外部请求）</footer>
</div>
<script>
const OPTIONS = {json.dumps(options)};
const TIMELINES = {json.dumps(timelines)};
function mount(id, option) {{
  const el = document.getElementById(id);
  if (!el || !option) return;
  const chart = echarts.init(el, null, {{renderer: "canvas"}});
  chart.setOption(option);
  window.addEventListener("resize", () => chart.resize());
  return chart;
}}
mount("c-ttft50", OPTIONS.ttft50);
mount("c-ttft90", OPTIONS.ttft90);
mount("c-throughput", OPTIONS.throughput);
mount("c-rps", OPTIONS.rps);
mount("c-e2e", OPTIONS.e2e);
mount("c-cache", OPTIONS.cache);
mount("c-cdf-ttft", OPTIONS.cdf && OPTIONS.cdf.ttft);
mount("c-cdf-e2e", OPTIONS.cdf && OPTIONS.cdf.e2e);

const keys = Object.keys(TIMELINES);
let timelineChart = null;
function show(key) {{
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.key === key));
  if (!timelineChart) timelineChart = mount("c-timeline", TIMELINES[key]);
  else timelineChart.setOption(TIMELINES[key], true);
}}
document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => show(t.dataset.key)));
if (keys.length) show(keys[0]);
{EXPLORER_JS.replace("__EXPLORER__", json.dumps(explorer))}
</script>
</body></html>"""


def build_report(root: Path, out: Path | None = None) -> Path:
    points, targets = discover(root)
    if not points:
        raise SystemExit(f"no completed AgentX points under {root}")
    echarts = ECHARTS_FILE.read_text(encoding="utf-8") if ECHARTS_FILE.exists() else ""
    html = render_html(points, targets, echarts)
    out = out or (root / "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    (root / "report_data.json").write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "targets": targets,
                "points": points,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="AgentX HTML report")
    p.add_argument("--root", default="/results")
    p.add_argument("--out", default="")
    args = p.parse_args()
    path = build_report(Path(args.root), Path(args.out) if args.out else None)
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
