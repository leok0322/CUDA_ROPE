#!/usr/bin/env bash

# 不使用 set -e：循环内编译或运行失败时继续测下一个配置，而非退出脚本
set -u
set -o pipefail

# 搜索空间：BLOCK_SIZE 必须是 32 的倍数（覆盖完整 warp），且 <= 1024（CUDA 每 block 线程数上限）
BS_VALUES=(32 64 128 256 512 1024)

# 切换到项目根目录（使脚本在任意目录下调用均正确）
cd "$(dirname "$0")"

COMMON="src/kernels/common.cuh"
OUTPUT="autotune/kernel_0_autotune_results.txt"

mkdir -p "$(dirname "$OUTPUT")"
echo "" > "$OUTPUT"

export DEVICE="0"

TOTAL_CONFIGS=${#BS_VALUES[@]}
CONFIG_NUM=0

for bs in "${BS_VALUES[@]}"; do
    echo ""
    CONFIG_NUM=$(( CONFIG_NUM + 1 ))

    config="BLOCK_SIZE=$bs"
    echo "($CONFIG_NUM/$TOTAL_CONFIGS): $config"

    # 原地替换 common.cuh 中的 BLOCK_SIZE 宏值
    # BLOCK_SIZE 是编译期常量，每次变化都需要完整重编译
    sed -i "s/#define BLOCK_SIZE .*/#define BLOCK_SIZE $bs/" "$COMMON"

    # 重新编译
    if ! cmake --build cmake-build-release --target validation -- -j 18 2>&1 | tee -a "$OUTPUT"; then
        echo "COMPILE FAILED: $config" | tee -a "$OUTPUT"
        continue
    fi

    echo "($CONFIG_NUM/$TOTAL_CONFIGS): $config" | tee -a "$OUTPUT"
    timeout -v 10 ./validation 0 2>&1 | tee -a "$OUTPUT"
    echo "-------------------" | tee -a "$OUTPUT"
    echo "" | tee -a "$OUTPUT"
done

# 恢复默认值，避免脚本结束后 common.cuh 停留在最后一个测试配置
sed -i "s/#define BLOCK_SIZE .*/#define BLOCK_SIZE 256/" "$COMMON"

echo ""
echo "Autotune complete. Results in $OUTPUT"
