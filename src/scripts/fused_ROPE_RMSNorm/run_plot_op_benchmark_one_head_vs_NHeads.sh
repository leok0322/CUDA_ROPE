#!/bin/bash

# run_plot_op_benchmark_one_head_vs_NHeads.sh —— 比较 one-head 算子和 NHeads 算子的 benchmark 结果。
#   默认读取：
#     benchmark_results/ROPE_python_op_benchmark_result_one_head.txt
#     benchmark_results/ROPE_python_op_benchmark_result_NHeads.txt
#   调用 plot_op_benchmark_one_head_vs_NHeads.py：解析两份结果 → 按 num_tokens 画 one-head op vs NHeads op
#   的对比折线，输出到 plot_output/one_head_vs_NHeads_{metric}.png。
#   plot 脚本一次只画一个 metric，故本 sh：未传 --metric → 默认画三张(gflops/time/speedup)；
#   传了 --metric X → 只画那一种。其余参数原样透传给 plot 脚本。
#
# 用法（参数透传给 plot_op_benchmark_one_head_vs_NHeads.py）：
#   ./run_plot_op_benchmark_one_head_vs_NHeads.sh
#   ./run_plot_op_benchmark_one_head_vs_NHeads.sh --metric time
#   ./run_plot_op_benchmark_one_head_vs_NHeads.sh --metric speedup
#   ./run_plot_op_benchmark_one_head_vs_NHeads.sh --one-head-file benchmark_results/ROPE_python_op_benchmark_result_one_head.txt
#   ./run_plot_op_benchmark_one_head_vs_NHeads.sh --nheads-file benchmark_results/ROPE_python_op_benchmark_result_NHeads.txt
#
# 前置：先分别生成 one-head 和 NHeads 两份 benchmark 结果文件。


# set -e : 任意命令返回非零退出码时立即终止脚本（errexit）
# set -u : 引用未定义变量时报错退出，而非静默展开为空字符串（nounset）
# set -o pipefail : 管道整体退出码 = 所有段中最坏的退出码
#   默认行为（无 pipefail）：管道退出码 = 最后一段的退出码
#     示例：cmd_fail | tee log → tee 成功(0) → 管道退出码=0 → set -e 不触发，cmd_fail 的失败被吞掉
#   加了 pipefail：cmd_fail(1) | tee(0) → 管道退出码=1 → set -e 触发，脚本终止
set -euo pipefail

# 本脚本位于 src/scripts/fused_ROPE_RMSNorm/ 下，上溯 3 级到项目根 CUDA_ROPE/。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/plot_op_benchmark_one_head_vs_NHeads.log"

mkdir -p "$PROJECT_ROOT/logs"

# 激活虚拟环境（与 run_check_op.sh / run_benchmark_op.sh 一致），使 matplotlib 可用。
source /home/liam/python_linux/python_venv/.venv/bin/activate

# "$@" 透传命令行参数（--one-head-file/--nheads-file 等）给 plot 脚本。
# plot 脚本一次只画一个 metric，故：
#   · 用户没传 --metric → 默认把 gflops / time / speedup 三张图都画出来；
#   · 用户传了 --metric X → 只画那一种。
# 2>&1 合并 stderr 到 stdout；tee 同时输出终端与日志文件。
if printf '%s\n' "$@" | grep -q -- '--metric'; then
    uv run python "$SCRIPT_DIR/plot_op_benchmark_one_head_vs_NHeads.py" "$@" 2>&1 | tee "$LOG_FILE"
else
    : > "$LOG_FILE"
    for m in gflops time speedup; do
        echo "==================== metric=$m ====================" | tee -a "$LOG_FILE"
        uv run python "$SCRIPT_DIR/plot_op_benchmark_one_head_vs_NHeads.py" --metric "$m" "$@" 2>&1 | tee -a "$LOG_FILE"
    done
fi
