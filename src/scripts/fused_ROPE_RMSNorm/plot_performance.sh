#!/bin/bash

# SCRIPT_DIR：脚本自身所在目录的绝对路径
#
#   $0                    脚本的调用路径，如 ./plot_performance.sh
#   dirname "$0"          取目录部分，如 . 或 /a/b
#   cd "$(dirname "$0")"  cd 进该目录（处理相对路径）
#   pwd                   打印当前目录的绝对路径
#
#   整体效果：无论从哪个目录调用此脚本，SCRIPT_DIR 始终是脚本文件所在的绝对路径。
# 本脚本位于 src/scripts/fused_ROPE_RMSNorm/ 下，上溯【3 级】到项目根 CUDA_ROPE/。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 激活虚拟环境，使 python / uv 使用 .venv 中的包
source /home/liam/python_linux/python_venv/.venv/bin/activate

# 2>&1        将 stderr 合并到 stdout（两路输出合为一路）
# tee         将合并后的输出同时写入终端和日志文件
# plot_performance.py 现与本 sh 同在 src/scripts/fused_ROPE_RMSNorm/ 下，故用 $SCRIPT_DIR（脚本自身目录）。
uv run python "$SCRIPT_DIR/plot_performance.py" "$@" 2>&1
