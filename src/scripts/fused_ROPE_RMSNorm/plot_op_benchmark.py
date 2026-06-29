#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""把 benchmark_results/python_op_benchmark_result.txt 画成折线图（自定义算子 vs eager）。

数据来源：benchmark_fused_qknorm_rope.py 追加写的结果文件，每次运行一段，行形如：
  [2026-06-28 08:28:03]  repeats=100 warmup=10  sweep: ...
    float16   hd=128 il=0 ts=128  heads=8/8/8  num_tok=128  op: median=0.11ms perf=16.3 GFLOPS  \
        eager: median=0.57ms perf=3.2 GFLOPS  eager/op=5.04x  correct=PASS

横轴 = num_tokens（规模轴）；纵轴按 --metric 选 GFLOPS / 耗时(ms) / 加速比。
按 (dtype, head_dim, interleave, num_heads) 分组：每组 op 用实线○、eager 用虚线△（同色）。

用法：
  uv run src/scripts/plot_op_benchmark.py                 # 最近一次运行，GFLOPS 对比
  uv run src/scripts/plot_op_benchmark.py --metric time   # 改画中位耗时(ms)
  uv run src/scripts/plot_op_benchmark.py --metric speedup
  uv run src/scripts/plot_op_benchmark.py --all-runs      # 汇总文件里的全部运行（按时间戳区分）
"""
import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # 无显示环境也能存图；savefig 不依赖 GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 本文件在 src/scripts/fused_ROPE_RMSNorm/，上溯【4 级 .parent】到项目根 CUDA_ROPE/。
ROOT = Path(__file__).parent.parent.parent.parent
# 默认读 benchmark 实际写入的结果文件（benchmark_fused_qknorm_rope.py 的 DEFAULT_RESULT_FILE）；
#   可经 --result-file 改读别的（如 autotune 的 ROPE_autotune_op_benchmark_result.txt）。
DEFAULT_RESULT_FILE = ROOT / "benchmark_results" / "ROPE_python_op_benchmark_result.txt"
OUTPUT_DIR = ROOT / "plot_output"

_HEADER_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+repeats=")
_ROW_RE = re.compile(
    r"(?P<dtype>\w+)\s+hd=(?P<hd>\d+)\s+il=(?P<il>\d+)\s+ts=(?P<tss>\S+)\s+"
    r"heads=(?P<hq>\d+)/(?P<hk>\d+)/(?P<hv>\d+)\s+num_tok=(?P<nt>\d+)\s+"
    r"op:\s*median=(?P<opm>[\d.]+)ms\s*perf=(?P<opg>[\d.]+)\s*GFLOPS\s+"
    r"eager:\s*median=(?P<egm>[\d.]+)ms\s*perf=(?P<egg>[\d.]+)\s*GFLOPS\s+"
    r"eager/op=(?P<sp>[\d.]+)x\s+correct=(?P<corr>\w+)"
)


def parse_records(path: Path) -> list[dict]:
    """解析结果文件，返回每条 OK 记录（带其所属运行的时间戳 ts）。"""
    if not path.exists():
        print(f"错误：未找到结果文件 {path}（先跑 ./run_benchmark_op.sh 生成）", file=sys.stderr)
        sys.exit(1)
    records, cur_ts = [], "?"
    for line in path.read_text().splitlines():
        h = _HEADER_RE.match(line.strip())
        if h:
            cur_ts = h.group("ts")
            continue
        m = _ROW_RE.search(line)
        if not m:
            continue
        records.append({
            "ts": cur_ts,
            "dtype": m.group("dtype"), "hd": int(m.group("hd")), "il": int(m.group("il")),
            "heads": f"{m.group('hq')}/{m.group('hk')}/{m.group('hv')}",
            "num_tok": int(m.group("nt")),
            "op_ms": float(m.group("opm")), "op_gf": float(m.group("opg")),
            "eager_ms": float(m.group("egm")), "eager_gf": float(m.group("egg")),
            "speedup": float(m.group("sp")), "correct": m.group("corr"),
        })
    if not records:
        print(f"错误：{path} 中没有可解析的 OK 结果行", file=sys.stderr)
        sys.exit(1)
    return records


def group_key(rec, multi_run: bool) -> tuple:
    k = (rec["dtype"], rec["hd"], rec["il"], rec["heads"])
    return (rec["ts"],) + k if multi_run else k


def group_label(rec, multi_run: bool) -> str:
    base = f"{rec['dtype']} hd{rec['hd']} il{rec['il']} {rec['heads']}"
    return f"{rec['ts']} | {base}" if multi_run else base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["gflops", "time", "speedup"], default="gflops",
                    help="纵轴指标：gflops(默认) / time(中位 ms) / speedup(eager/op)")
    ap.add_argument("--all-runs", action="store_true",
                    help="画汇总文件里的全部运行（按时间戳区分）；缺省=只画最近一次")
    ap.add_argument("--result-file", dest="result_file", default=str(DEFAULT_RESULT_FILE),
                    help=f"要画的结果文件（缺省={DEFAULT_RESULT_FILE}）")
    args = ap.parse_args()

    records = parse_records(Path(args.result_file))
    if not args.all_runs:                       # 默认只取最近一次运行（时间戳最大）
        latest = max(r["ts"] for r in records)
        records = [r for r in records if r["ts"] == latest]
        run_note = f"latest run @ {latest}"
    else:
        run_note = "all runs"

    # 按组聚合 → 每组按 num_tok 排序
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(group_key(r, args.all_runs), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r["num_tok"])

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [plt.cm.tab10(i % 10) for i in range(len(groups))]
    single = len(groups) == 1
    all_nt = sorted({r["num_tok"] for r in records})

    for i, (key, recs) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        xs = [r["num_tok"] for r in recs]
        lbl = group_label(recs[0], args.all_runs)
        c = colors[i]
        if args.metric == "gflops":
            ax.plot(xs, [r["op_gf"] for r in recs], marker="o", linestyle="-",
                    color=c, linewidth=1.8, markersize=6, label=("op" if single else f"{lbl} [op]"))
            ax.plot(xs, [r["eager_gf"] for r in recs], marker="^", linestyle="--",
                    color=c, linewidth=1.5, markersize=6, label=("eager" if single else f"{lbl} [eager]"))
            # 在 op 点上标注加速比
            for r in recs:
                ax.annotate(f"{r['speedup']:.1f}x", xy=(r["num_tok"], r["op_gf"]),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=7, color=c)
        elif args.metric == "time":
            ax.plot(xs, [r["op_ms"] for r in recs], marker="o", linestyle="-",
                    color=c, linewidth=1.8, markersize=6, label=("op" if single else f"{lbl} [op]"))
            ax.plot(xs, [r["eager_ms"] for r in recs], marker="^", linestyle="--",
                    color=c, linewidth=1.5, markersize=6, label=("eager" if single else f"{lbl} [eager]"))
        else:  # speedup
            ax.plot(xs, [r["speedup"] for r in recs], marker="D", linestyle="-",
                    color=c, linewidth=2.0, markersize=6, label=(lbl if not single else "eager/op"))

    ylabel = {"gflops": "Performance (GFLOPS)", "time": "Median latency (ms)",
              "speedup": "Speedup (eager / op)"}[args.metric]
    title = {"gflops": "Fused QK-Norm+RoPE: op vs eager — GFLOPS",
             "time": "Fused QK-Norm+RoPE: op vs eager — latency",
             "speedup": "Fused QK-Norm+RoPE: speedup (eager / op)"}[args.metric]
    ax.set_xlabel("num_tokens", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}   ({run_note})", fontsize=13)
    ax.set_xscale("log", base=2)                 # num_tokens 跨度大(128..8192)，对数刻度更均匀
    ax.set_xticks(all_nt)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:.2f}"))
    if args.metric == "speedup":
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)   # 1x 基准线
    ax.legend(loc="best", fontsize=9, ncol=1 if single else 2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = OUTPUT_DIR / f"op_benchmark_{args.metric}.png"
    plt.savefig(out, dpi=150)
    print(f"已保存：{out}")


if __name__ == "__main__":
    main()
