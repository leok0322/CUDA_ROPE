#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""比较两个 NHeads 算法 benchmark 结果并画折线图。

默认读取：
  benchmark_results/ROPE_python_op_benchmark_result_NHeads_algorithm0.txt
  benchmark_results/ROPE_python_op_benchmark_result_NHeads_algorithm1.txt

两份文件格式与 benchmark_fused_qknorm_rope.py 追加写出的结果一致，行形如：
  [2026-06-28 08:28:03]  repeats=100 warmup=10  sweep: ...
    float16   hd=128 il=0 ts=128  heads=8/8/8  num_tok=128  op: median=0.11ms perf=16.3 GFLOPS  \
        eager: median=0.57ms perf=3.2 GFLOPS  eager/op=5.04x  correct=PASS

横轴 = num_tokens。纵轴按 --metric 选择：
  gflops  : 两个 NHeads op 的 GFLOPS
  time    : 两个 NHeads op 的中位耗时(ms)
  speedup : B 相对 A 的加速比，定义为 A_ms / B_ms

用法：
  uv run src/scripts/fused_ROPE_RMSNorm/plot_op_benchmark_NHeads_algorithm_compare.py
  uv run src/scripts/fused_ROPE_RMSNorm/plot_op_benchmark_NHeads_algorithm_compare.py --metric time
  uv run src/scripts/fused_ROPE_RMSNorm/plot_op_benchmark_NHeads_algorithm_compare.py --metric speedup

如果需要比较任意两个文件：
  uv run src/scripts/fused_ROPE_RMSNorm/plot_op_benchmark_NHeads_algorithm_compare.py \
    --a-file benchmark_results/ROPE_python_op_benchmark_result_NHeads_algorithm0.txt \
    --b-file benchmark_results/ROPE_python_op_benchmark_result_NHeads_algorithm0.txt \
    --a-label "NHeads A0" \
    --b-label "NHeads A0 copy"
"""
import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能存图；savefig 不依赖 GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# 本文件在 src/scripts/fused_ROPE_RMSNorm/，上溯 4 级 .parent 到项目根 CUDA_ROPE/。
ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_A_FILE = ROOT / "benchmark_results" / "ROPE_python_op_benchmark_result_NHeads_algorithm0.txt"
DEFAULT_B_FILE = ROOT / "benchmark_results" / "ROPE_python_op_benchmark_result_NHeads_algorithm1.txt"
OUTPUT_DIR = ROOT / "plot_output"

_HEADER_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+repeats=")
_ROW_RE = re.compile(
    r"(?P<dtype>\w+)\s+hd=(?P<hd>\d+)\s+il=(?P<il>\d+)\s+ts=(?P<tss>\S+)\s+"
    r"heads=(?P<hq>\d+)/(?P<hk>\d+)/(?P<hv>\d+)\s+num_tok=(?P<nt>\d+)\s+"
    r"op:\s*median=(?P<opm>[\d.]+)ms\s*perf=(?P<opg>[\d.]+)\s*GFLOPS\s+"
    r"eager:\s*median=(?P<egm>[\d.]+)ms\s*perf=(?P<egg>[\d.]+)\s*GFLOPS\s+"
    r"eager/op=(?P<sp>[\d.]+)x\s+correct=(?P<corr>\w+)"
)


def resolve_path(path_text: str) -> Path:
    """支持传入绝对路径，也支持传入相对项目根目录的路径。"""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_records(path: Path, label: str) -> list[dict]:
    """解析结果文件，返回每条 benchmark 记录。"""
    if not path.exists():
        print(f"错误：未找到 {label} 结果文件 {path}", file=sys.stderr)
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
            "dtype": m.group("dtype"),
            "hd": int(m.group("hd")),
            "il": int(m.group("il")),
            "heads": f"{m.group('hq')}/{m.group('hk')}/{m.group('hv')}",
            "num_tok": int(m.group("nt")),
            "op_ms": float(m.group("opm")),
            "op_gf": float(m.group("opg")),
            "eager_ms": float(m.group("egm")),
            "eager_gf": float(m.group("egg")),
            "eager_speedup": float(m.group("sp")),
            "correct": m.group("corr"),
        })

    if not records:
        print(f"错误：{path} 中没有可解析的 benchmark 结果行", file=sys.stderr)
        sys.exit(1)
    return records


def latest_run(records: list[dict]) -> tuple[str, list[dict]]:
    """选取时间戳最大的那次运行。"""
    latest = max(r["ts"] for r in records)
    return latest, [r for r in records if r["ts"] == latest]


def record_key(rec: dict) -> tuple:
    return (rec["dtype"], rec["hd"], rec["il"], rec["heads"], rec["num_tok"])


def group_key_from_key(key: tuple) -> tuple:
    dtype, hd, il, heads, _num_tok = key
    return dtype, hd, il, heads


def group_label(group_key: tuple) -> str:
    dtype, hd, il, heads = group_key
    return f"{dtype} hd{hd} il{il} {heads}"


def index_records(records: list[dict], label: str) -> dict[tuple, dict]:
    """按可比较配置建索引；同一配置重复时保留最后一条并提示。"""
    indexed: dict[tuple, dict] = {}
    duplicate_count = 0
    for rec in records:
        key = record_key(rec)
        if key in indexed:
            duplicate_count += 1
        indexed[key] = rec

    if duplicate_count:
        print(f"警告：{label} 有 {duplicate_count} 条重复配置，已保留每组最后一条", file=sys.stderr)
    return indexed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["gflops", "time", "speedup"], default="gflops",
                    help="纵轴指标：gflops(默认) / time(中位 ms) / speedup(B/A)")
    ap.add_argument("--a-file", default=str(DEFAULT_A_FILE),
                    help=f"A benchmark 结果文件（缺省={DEFAULT_A_FILE}）")
    ap.add_argument("--b-file", default=str(DEFAULT_B_FILE),
                    help=f"B benchmark 结果文件（缺省={DEFAULT_B_FILE}）")
    ap.add_argument("--a-label", default="NHeads algorithm0",
                    help="图例中的 A 名称")
    ap.add_argument("--b-label", default="NHeads algorithm1",
                    help="图例中的 B 名称")
    args = ap.parse_args()

    a_ts, a_records = latest_run(parse_records(resolve_path(args.a_file), args.a_label))
    b_ts, b_records = latest_run(parse_records(resolve_path(args.b_file), args.b_label))

    a_by_key = index_records(a_records, args.a_label)
    b_by_key = index_records(b_records, args.b_label)
    common_keys = sorted(set(a_by_key) & set(b_by_key), key=str)

    if not common_keys:
        print("错误：两个结果文件没有可比较的共同配置", file=sys.stderr)
        print(f"{args.a_label} latest run: {a_ts}, records={len(a_by_key)}", file=sys.stderr)
        print(f"{args.b_label} latest run: {b_ts}, records={len(b_by_key)}", file=sys.stderr)
        sys.exit(1)

    missing_in_b = len(set(a_by_key) - set(b_by_key))
    missing_in_a = len(set(b_by_key) - set(a_by_key))
    if missing_in_b or missing_in_a:
        print(
            f"提示：仅绘制共同配置；{args.a_label} 独有 {missing_in_b} 条，"
            f"{args.b_label} 独有 {missing_in_a} 条",
            file=sys.stderr,
        )

    groups: dict[tuple, list[tuple[dict, dict]]] = {}
    for key in common_keys:
        groups.setdefault(group_key_from_key(key), []).append((a_by_key[key], b_by_key[key]))
    for pairs in groups.values():
        pairs.sort(key=lambda pair: pair[0]["num_tok"])

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [plt.cm.tab10(i % 10) for i in range(len(groups))]
    single = len(groups) == 1
    all_nt = sorted({a["num_tok"] for pairs in groups.values() for a, _b in pairs})

    for i, (gkey, pairs) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
        xs = [a["num_tok"] for a, _b in pairs]
        lbl = group_label(gkey)
        c = colors[i]

        if args.metric == "gflops":
            ax.plot(xs, [a["op_gf"] for a, _b in pairs], marker="o", linestyle="-",
                    color=c, linewidth=1.8, markersize=6,
                    label=(args.a_label if single else f"{lbl} [{args.a_label}]"))
            ax.plot(xs, [b["op_gf"] for _a, b in pairs], marker="^", linestyle="--",
                    color=c, linewidth=1.5, markersize=6,
                    label=(args.b_label if single else f"{lbl} [{args.b_label}]"))
            for a, b in pairs:
                speedup = a["op_ms"] / b["op_ms"] if b["op_ms"] > 0 else float("inf")
                ax.annotate(f"{speedup:.2f}x", xy=(a["num_tok"], b["op_gf"]),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=7, color=c)

        elif args.metric == "time":
            ax.plot(xs, [a["op_ms"] for a, _b in pairs], marker="o", linestyle="-",
                    color=c, linewidth=1.8, markersize=6,
                    label=(args.a_label if single else f"{lbl} [{args.a_label}]"))
            ax.plot(xs, [b["op_ms"] for _a, b in pairs], marker="^", linestyle="--",
                    color=c, linewidth=1.5, markersize=6,
                    label=(args.b_label if single else f"{lbl} [{args.b_label}]"))

        else:  # speedup
            speedups = [a["op_ms"] / b["op_ms"] for a, b in pairs]
            ax.plot(xs, speedups, marker="D", linestyle="-",
                    color=c, linewidth=2.0, markersize=6,
                    label=(lbl if not single else f"{args.b_label} / {args.a_label}"))

    ylabel = {
        "gflops": "Performance (GFLOPS)",
        "time": "Median latency (ms)",
        "speedup": f"Speedup ({args.a_label} ms / {args.b_label} ms)",
    }[args.metric]
    title = {
        "gflops": "Fused QK-Norm+RoPE: NHeads algorithm compare — GFLOPS",
        "time": "Fused QK-Norm+RoPE: NHeads algorithm compare — latency",
        "speedup": "Fused QK-Norm+RoPE: NHeads algorithm speedup",
    }[args.metric]

    ax.set_xlabel("num_tokens", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}   ({args.a_label} @ {a_ts}; {args.b_label} @ {b_ts})", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_nt)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:.2f}"))
    if args.metric == "speedup":
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.legend(loc="best", fontsize=9, ncol=1 if single else 2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = OUTPUT_DIR / f"NHeads_algorithm_compare_{args.metric}.png"
    plt.savefig(out, dpi=150)
    print(f"已保存：{out}")


if __name__ == "__main__":
    main()
