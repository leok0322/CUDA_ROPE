#!/bin/bash

# run_plot_op.sh —— 把 benchmark_results/python_op_benchmark_result.txt 画成折线图。
#   调用 src/scripts/plot_op_benchmark.py：解析 benchmark 结果 → 按 num_tokens 画
#   自定义算子 vs eager 的对比折线，输出到 plot_output/op_benchmark_{metric}.png。
#
# 透传参数给 plot 脚本，例如：
#   ./run_plot_op.sh                      # GFLOPS 对比（最近一次运行）
#   ./run_plot_op.sh --metric time        # 改画中位耗时(ms)
#   ./run_plot_op.sh --metric speedup      # 画加速比(eager/op)
#   ./run_plot_op.sh --all-runs            # 画汇总文件里的全部运行
#
# 前置：先跑 ./run_benchmark_op.sh 生成 benchmark_results/python_op_benchmark_result.txt。


# set -e : 任意命令返回非零退出码时立即终止脚本（errexit）
# set -u : 引用未定义变量时报错退出，而非静默展开为空字符串（nounset）
# set -o pipefail : 管道整体退出码 = 所有段中最坏的退出码
#   默认行为（无 pipefail）：管道退出码 = 最后一段的退出码
#     示例：cmd_fail | tee log → tee 成功(0) → 管道退出码=0 → set -e 不触发，cmd_fail 的失败被吞掉
#   加了 pipefail：cmd_fail(1) | tee(0) → 管道退出码=1 → set -e 触发，脚本终止
set -euo pipefail

# 本脚本位于 src/scripts/fused_ROPE_RMSNorm/ 下，上溯【3 级】到项目根 CUDA_ROPE/。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/plot_op_benchmark.log"

mkdir -p "$PROJECT_ROOT/logs"

# 激活虚拟环境（与 run_check_op.sh / run_benchmark_op.sh 一致），使 matplotlib 可用。
source /home/liam/python_linux/python_venv/.venv/bin/activate

# "$@" 透传命令行参数（--metric/--all-runs）给 plot 脚本。
# 2>&1 合并 stderr 到 stdout；tee 同时输出终端与日志文件。
# plot 脚本现与本 sh 同在 src/scripts/fused_ROPE_RMSNorm/ 下，故用 $SCRIPT_DIR（脚本自身目录）。
uv run python "$SCRIPT_DIR/plot_op_benchmark.py" "$@" 2>&1 | tee "$LOG_FILE"
