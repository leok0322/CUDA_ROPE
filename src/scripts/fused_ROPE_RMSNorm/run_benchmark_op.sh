#!/bin/bash

# run_benchmark_op.sh —— 运行 src/scripts/benchmark_fused_qknorm_rope.py，
#   对【自定义融合算子(torch.compile 替换出的融合 kernel) vs eager(分步子图)】做耗时对比。
#   逐配置(dtype × head_dim × interleave × token_size × num_heads)各跑一次：
#   CUDA Event 计时(WARMUP 预热 + REPEATS 次)，先 allclose 校验数值一致，再比中位数耗时。
#   终端打印对比表；结果追加写入 benchmark_results/python_op_benchmark_result.txt。
#
# 可加参数透传给 benchmark 脚本，例如：
#   ./run_benchmark_op.sh --token-sizes 256 1024 4096       # 按 num_tokens 规模扫描
#   ./run_benchmark_op.sh --dtype float16 bfloat16 --head-dim 128
#
# 前置：需先用 CMake 构建出 ROPE_cuda.*.so（项目根或 cmake-build*/）；改了 kernel/dispatch 后
#   务必【重新构建 .so】再跑，否则测到的是旧 kernel 的耗时。需 GPU（CUDA Event 计时）。

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
LOG_FILE="$PROJECT_ROOT/logs/benchmark_fused_qknorm_rope.log"

mkdir -p "$PROJECT_ROOT/logs"

# 激活虚拟环境（与 run_check_op.sh / run_visualization.sh 一致）。
source /home/liam/python_linux/python_venv/.venv/bin/activate

# "$@" 透传命令行参数给 benchmark 脚本（如 --token-sizes/--dtype/--head-dim）。
# 2>&1 合并 stderr 到 stdout；tee 同时输出终端与日志文件。
# benchmark 脚本现与本 sh 同在 src/scripts/fused_ROPE_RMSNorm/ 下，故用 $SCRIPT_DIR（脚本自身目录）。
uv run python "$SCRIPT_DIR/benchmark_fused_qknorm_rope.py" "$@" 2>&1 | tee "$LOG_FILE"
