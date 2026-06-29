#!/bin/bash

# run_check_op.sh —— 运行 src/scripts/test_fused_qknorm_rope.py，多维度扫描校验自定义融合算子。
#   该脚本复用 src/app/runner.py 的 Runner，对【dtype × head_dim × interleave】各组合各跑一次
#   （每组合独立子进程，避免 torch.compile/pattern 状态串扰）：eager 前向得参考 y_ref →
#   torch.compile 把 "QK-Norm + RoPE" 子图替换成自定义算子 → 比对 y 与 y_ref（allclose）。
#   终端/日志逐组合打印 ✅PASS / ❌FAIL / ⏭ SKIP，并给出汇总；有 FAIL/ERROR 时退出码非 0。
#
# 可加参数过滤子集（透传给 test 脚本），例如：
#   ./run_check_op.sh --dtype float16 --head-dim 128            # 只测 fp16 × 128
#   ./run_check_op.sh --interleave 1                            # 只测 interleave 风格
#
# 前置：需先用 CMake 构建出 ROPE_cuda.*.so 并置于项目根目录（worker 会 import 它）；否则各组合
#   打印 ⏭ SKIP "ROPE_cuda 未加载"。改了 kernel/dispatch 后务必【重新构建 .so】再跑，避免用到旧库。

# set -e : 任意命令返回非零退出码时立即终止脚本（errexit）
# set -u : 引用未定义变量时报错退出，而非静默展开为空字符串（nounset）
# set -o pipefail : 管道整体退出码 = 所有段中最坏的退出码
#   默认行为（无 pipefail）：管道退出码 = 最后一段的退出码
#     示例：cmd_fail | tee log → tee 成功(0) → 管道退出码=0 → set -e 不触发，cmd_fail 的失败被吞掉
#   加了 pipefail：cmd_fail(1) | tee(0) → 管道退出码=1 → set -e 触发，脚本终止
set -euo pipefail


# 本脚本位于 src/scripts/fused_ROPE_RMSNorm/ 下，需上溯【3 级】到项目根 CUDA_ROPE/。
#   SCRIPT_DIR  = 脚本自身目录；PROJECT_ROOT = 项目根（所有项目相对路径都基于它）。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/check_fused_qknorm_rope.log"

mkdir -p "$PROJECT_ROOT/logs"

# 激活虚拟环境，使 python / uv 使用 .venv 中的包（与 run_visualization.sh 一致）。
source /home/liam/python_linux/python_venv/.venv/bin/activate

# test 脚本内部自行把 src/app 与项目根加入 sys.path（import util/runner 与 ROPE_cuda.so）；
#   "$@" 透传命令行参数（如 --dtype/--head-dim/--interleave）给 test 脚本做子集过滤。
# 2>&1  将 stderr 合并到 stdout；tee 同时输出到终端与日志文件。
# test 脚本现与本 sh 同在 src/scripts/fused_ROPE_RMSNorm/ 下，故用 $SCRIPT_DIR（脚本自身目录）。
uv run python "$SCRIPT_DIR/test_fused_qknorm_rope.py" "$@" 2>&1 | tee "$LOG_FILE"
