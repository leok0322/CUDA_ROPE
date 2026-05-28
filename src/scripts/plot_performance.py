#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""
用法：
  uv run src/scripts/plot_performance.py              # 绘制默认方阵 + 非方阵维度
  uv run src/scripts/plot_performance.py 128 512      # 只绘制指定方阵维度
  uv run src/scripts/plot_performance.py --all-sizes  # 绘制全部出现过的维度
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).parent.parent.parent          # 项目根目录
RESULTS_DIR = ROOT / "benchmark_results"
KERNELS_DIR = ROOT / "src" / "kernels"

DEFAULT_SQUARE     = [128, 256, 512, 1024, 2048, 4096]
DEFAULT_NONSQUARE  = [(128, 4096), (256, 4096), (4096, 128), (4096, 1024)]


def discover_kernel_labels() -> dict[int, str]:
    """从 src/kernels/ 文件名解析 kernel id → 标签，忽略 common.cuh。"""
    labels: dict[int, str] = {}
    for f in sorted(KERNELS_DIR.glob("*.cuh")):
        m = re.match(r"^(\d+)_(.+)\.cuh$", f.name)
        if not m:
            continue
        kid = int(m.group(1))
        # 去掉 ROPE_kernel_ 前缀，保留关键词
        name = re.sub(r"^ROPE_kernel_", "", m.group(2))
        labels[kid] = f"K{kid} {name}"
    return labels


def parse_result(kernel_id: int) -> dict[tuple[int, int], float]:
    """解析 benchmark_results/ROPE_kernel_{id}_result.txt，返回 {(m,n): gflops}。"""
    path = RESULTS_DIR / f"ROPE_kernel_{kernel_id}_result.txt"
    if not path.exists():
        return {}
    pattern = re.compile(
        r"performance:\s*\(([\d.]+)\)\s*GFLOPS\.\s*size:\s*\((\d+),\s*(\d+)\)"
    )
    results: dict[tuple[int, int], float] = {}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            results[(int(m.group(2)), int(m.group(3)))] = float(m.group(1))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dims", nargs="*", type=int, metavar="DIM",
                        help="方阵维度 m=n，不填则使用默认维度")
    parser.add_argument("--all-sizes", action="store_true",
                        help="自动绘制所有出现过的维度")
    args = parser.parse_args()

    # ── 发现 kernel 标签 ──────────────────────────────────────────────────────
    kernel_labels = discover_kernel_labels()
    if not kernel_labels:
        print("错误：src/kernels/ 下未找到 *.cuh 文件", file=sys.stderr)
        sys.exit(1)
    kernel_ids = sorted(kernel_labels)

    # ── 加载所有 kernel 数据 ──────────────────────────────────────────────────
    data: dict[int, dict[tuple[int, int], float]] = {}
    for kid in kernel_ids:
        parsed = parse_result(kid)
        if parsed:
            data[kid] = parsed
        else:
            print(f"警告：未找到 kernel_{kid} 结果文件，跳过", file=sys.stderr)

    if not data:
        print("错误：benchmark_results/ 下没有找到任何结果文件", file=sys.stderr)
        sys.exit(1)

    # ── 确定所有 size ─────────────────────────────────────────────────────────
    all_sizes: set[tuple[int, int]] = set()
    for d in data.values():
        all_sizes.update(d.keys())

    if args.all_sizes:
        square_dims = sorted({r for r, c in all_sizes if r == c})
        nonsquare   = sorted((r, c) for r, c in all_sizes if r != c)
    elif args.dims:
        square_dims = sorted(set(args.dims))
        nonsquare   = DEFAULT_NONSQUARE
    else:
        square_dims = DEFAULT_SQUARE
        nonsquare   = DEFAULT_NONSQUARE

    plot_sizes = [(d, d) for d in square_dims] + [s for s in nonsquare if s in all_sizes]

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    sq_colors  = [plt.cm.tab10(i)  for i in range(len(square_dims))]
    nsq_colors = [plt.cm.Set2(i)   for i in range(len(nonsquare))]

    x_positions = kernel_ids  # kernel id 作为横坐标

    # 方阵折线（实线 + 圆形标记）
    for i, dim in enumerate(square_dims):
        key = (dim, dim)
        # 缺失维度记为 0
        ys = [data.get(kid, {}).get(key, 0.0) for kid in kernel_ids]
        ax.plot(x_positions, ys,
                label=f"{dim}×{dim}",
                marker="o", linewidth=1.8, markersize=5,
                color=sq_colors[i])

    # 非方阵折线（点划线 + 三角形标记）
    for i, (rows, cols) in enumerate(nonsquare):
        if (rows, cols) not in all_sizes:
            print(f"警告：{rows}×{cols} 在所有 kernel 中均无数据，跳过", file=sys.stderr)
            continue
        key = (rows, cols)
        ys = [data.get(kid, {}).get(key, 0.0) for kid in kernel_ids]
        ax.plot(x_positions, ys,
                label=f"{rows}×{cols}",
                marker="^", linewidth=1.5, markersize=5,
                color=nsq_colors[i], linestyle="-.")

    # 平均线（所有绘制维度的平均 GFLOPS）
    avg_ys = []
    for kid in kernel_ids:
        vals = [data.get(kid, {}).get(key, 0.0) for key in plot_sizes]
        avg_ys.append(sum(vals) / len(vals) if vals else 0.0)

    ax.plot(x_positions, avg_ys,
            label="Average", marker="D", linewidth=2.5, markersize=7,
            color="black", linestyle="--", zorder=5)
    for x, y in zip(x_positions, avg_ys):
        ax.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", fontsize=7, color="black", zorder=6)

    # ── 坐标轴 / 图例 ─────────────────────────────────────────────────────────
    ax.set_xlabel("Kernel", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title("CUDA ROPE Kernel Performance  (○ square  △ non-square)", fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([kernel_labels[k] for k in kernel_ids],
                       rotation=25, ha="right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = ROOT / "performance_plot.png"
    plt.savefig(out, dpi=150)
    print(f"已保存：{out}")
    plt.show()


if __name__ == "__main__":
    main()
