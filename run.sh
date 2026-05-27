#!/usr/bin/env bash
# set -e : 任意命令返回非零退出码时立即终止脚本（errexit）
# set -u : 引用未定义变量时报错退出，而非静默展开为空字符串（nounset）
# set -o pipefail : 管道整体退出码 = 所有段中最坏的退出码
#   默认行为（无 pipefail）：管道退出码 = 最后一段的退出码
#     示例：cmd_fail | tee log → tee 成功(0) → 管道退出码=0 → set -e 不触发，cmd_fail 的失败被吞掉
#   加了 pipefail：cmd_fail(1) | tee(0) → 管道退出码=1 → set -e 触发，脚本终止
set -euo pipefail

# 可执行文件路径（CMakeLists.txt 中 add_executable 的输出路径）
EXEC="./validation"
# 日志目录
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

for i in $(seq 0 0); do
    LOG_FILE="$LOG_DIR/kernel_${i}.log"
    echo "Running kernel ${i} → ${LOG_FILE}"
    # ── tee 方案：同时输出到 terminal 和文件 ────────────────────────────────
    # 重定向从左到右依次执行，"|" 先把 fd1(stdout) 接到管道入口
    # 2>&1 在管道符之前：将 fd2(stderr) 也指向 fd1 此刻所指（管道入口）
    # → stdout + stderr 合并为一路进入管道 → tee 分发到 文件 + terminal
    "$EXEC" "$i" 2>&1 | tee "$LOG_FILE"

    # ── 仅写文件方案（不输出到 terminal）参考 ───────────────────────────────
    # 重定向顺序必须是 > file 在前，2>&1 在后：
    #   步骤1: > "$LOG_FILE"  → fd1(stdout) 指向文件
    #   步骤2: 2>&1           → fd2(stderr) 指向 fd1 此刻所指 → 文件
    # 错误写法 2>&1 > "$LOG_FILE"：
    #   步骤1: 2>&1           → fd2 指向 fd1 此刻所指 → terminal（尚未重定向）
    #   步骤2: > "$LOG_FILE"  → fd1 指向文件，但 fd2 已固定在 terminal
    #   结果：stdout 进文件，stderr 仍在 terminal
    # "$EXEC" "$i" > "$LOG_FILE" 2>&1
done

echo "All kernels done. Logs in ${LOG_DIR}/"
