#!/bin/bash

# run_plot_op.sh —— 把 benchmark 结果（默认 benchmark_results/ROPE_python_op_benchmark_result.txt）
#   画成折线图。调用 plot_op_benchmark.py：解析结果 → 按 num_tokens 画自定义算子 vs eager 的
#   对比折线，输出到 plot_output/op_benchmark_{metric}.png。
#   plot 脚本【一次只画一个 metric】，故本 sh：未传 --metric → 默认画三张(gflops/time/speedup)；
#   传了 --metric X → 只画那一种。其余参数原样透传给 plot 脚本，由其据参数画图。
#
# 用法（参数透传给 plot_op_benchmark.py）：
#   ./run_plot_op.sh                       # 默认：gflops/time/speedup 三张都画（最近一次运行）
#   ./run_plot_op.sh --metric time         # 只画中位耗时(ms) 一张
#   ./run_plot_op.sh --metric speedup       # 只画加速比(eager/op) 一张
#   ./run_plot_op.sh --all-runs            # 画汇总文件里的全部运行（仍是三张；未限定 metric）
#   ./run_plot_op.sh --result-file benchmark_results/ROPE_autotune_op_benchmark_result.txt   # 换数据源
#
# 前置：先跑 ./run_benchmark_op.sh 生成 benchmark_results/ROPE_python_op_benchmark_result.txt。


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

# "$@" 透传命令行参数（--all-runs/--result-file 等）给 plot 脚本。
# plot 脚本一次只画一个 metric，故：
#   · 用户【没传 --metric】→ 默认把 gflops / time / speedup 三张图都画出来；
#   · 用户【传了 --metric X】→ 只画那一种（原样透传）。
# 2>&1 合并 stderr 到 stdout；tee 同时输出终端与日志文件。
# plot 脚本现与本 sh 同在 src/scripts/fused_ROPE_RMSNorm/ 下，故用 $SCRIPT_DIR（脚本自身目录）。
if printf '%s\n' "$@" | grep -q -- '--metric'; then
    uv run python "$SCRIPT_DIR/plot_op_benchmark.py" "$@" 2>&1 | tee "$LOG_FILE"
else
    : > "$LOG_FILE"
    for m in gflops time speedup; do
        echo "==================== metric=$m ====================" | tee -a "$LOG_FILE"
        uv run python "$SCRIPT_DIR/plot_op_benchmark.py" --metric "$m" "$@" 2>&1 | tee -a "$LOG_FILE"
    done
fi
