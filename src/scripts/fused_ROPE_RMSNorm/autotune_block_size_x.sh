#!/usr/bin/env bash
#
# autotune_block_size_x.sh —— 微调 src/kernels/common.cuh 的 BLOCK_SIZE_X，找最佳性能。
#   对每个候选值：sed 改宏 → 重编译 ROPE_cuda(.so) → 跑 benchmark_fused_qknorm_rope.py
#   (固定单配置) → 解析 op 的 GFLOPS → 取最大者为最佳。
#
# BLOCK_SIZE_X 约束：必须是 32 的倍数(一个 warp 负责一行，static_assert BLOCK_SIZE_X%32==0)，
#   且 ≤ 1024(CUDA 每 block 线程数上限)。它决定每块 warp 数 = BLOCK_SIZE_X/32，影响占用率。
#
# 用法：./autotune_block_size_x.sh
set -u
set -o pipefail

# 本脚本位于 src/scripts/fused_ROPE_RMSNorm/ 下，cd 到项目根(上溯 3 级)，
#   使下面 src/kernels、cmake-build-release、benchmark_results、autotune 等相对路径都基于项目根。
cd "$(dirname "$0")/../../.." || exit 1
SCRIPT_DIR="$(pwd)"      # = 项目根 CUDA_ROPE/

# ── 搜索空间 & 固定 benchmark 配置（单配置便于横向比较；可按需改）──────────────────
BSX_VALUES=(32 64 128 256 512 1024)
BENCH_PY="src/scripts/fused_ROPE_RMSNorm/benchmark_fused_qknorm_rope.py"   # py 已移到本子文件夹
# 固定一个较大规模的单配置，信号干净：fp16 × head_dim128 × neox × num_tokens=4096 × (8,8,8)
BENCH_ARGS=(--dtype float16 --head-dim 128 --interleave 0 --token-sizes 4096 --num-heads 8,8,8)

COMMON="src/kernels/common.cuh"
BUILD_DIR="cmake-build-release"
RESULT_FILE="benchmark_results/ROPE_autotune_op_benchmark_result.txt"
OUTPUT="autotune/ROPE_autotune_block_size_x_autotune_results.txt"
mkdir -p "$(dirname "$OUTPUT")"
: > "$OUTPUT"

export DEVICE="0"
JOBS="$(nproc)"

# ── 记下原始值，结束/中断时恢复（避免 common.cuh 停在最后一个测试配置）──────────────
ORIG_BSX="$(grep -oP '^#define BLOCK_SIZE_X \K[0-9]+' "$COMMON" | head -1)"
restore() { sed -i "s/^#define BLOCK_SIZE_X .*/#define BLOCK_SIZE_X ${ORIG_BSX}/" "$COMMON"; }
trap restore EXIT INT TERM
echo "原始 BLOCK_SIZE_X = ${ORIG_BSX}（结束后会恢复）" | tee -a "$OUTPUT"

# ── 激活虚拟环境（benchmark 需 torch / ROPE_cuda）──────────────────────────────────
source /home/liam/python_linux/python_venv/.venv/bin/activate

best_gf="0"; best_bsx=""
TOTAL=${#BSX_VALUES[@]}; idx=0

for bsx in "${BSX_VALUES[@]}"; do
  idx=$(( idx + 1 ))
  echo "" | tee -a "$OUTPUT"
  echo "==================== ($idx/$TOTAL) BLOCK_SIZE_X=$bsx ====================" | tee -a "$OUTPUT"

  # 约束检查（候选已满足，留作防御）
  if (( bsx % 32 != 0 || bsx > 1024 )); then
    echo "SKIP: BLOCK_SIZE_X=$bsx 不满足 (32 的倍数 且 ≤1024)" | tee -a "$OUTPUT"
    continue
  fi

  # 1) 改宏（BLOCK_SIZE_X 是编译期常量，必须重编译才生效）
  sed -i "s/^#define BLOCK_SIZE_X .*/#define BLOCK_SIZE_X $bsx/" "$COMMON"

  # 2) 重编译 .so
  if ! cmake --build "$BUILD_DIR" --target ROPE_cuda -- -j "$JOBS" >>"$OUTPUT" 2>&1; then
    echo "COMPILE FAILED: BLOCK_SIZE_X=$bsx" | tee -a "$OUTPUT"
    continue
  fi

  # 3) 跑 benchmark（单配置），用结果文件行数变化判定本次是否产出 OK
  #    benchmark 会把结构化结果【追加】到 $RESULT_FILE(benchmark_results/...txt)；
  #    该文件累积多次运行，故不能无脑取最后一行——需用"前后计数差"确认本次确有新 OK。
  #
  # before：跑之前先数 $RESULT_FILE 里 'op: median=' 行数(=已有的 OK 结果条数)。
  #   grep -c        ：只输出匹配行【条数】；
  #   2>/dev/null    ：丢弃 stderr(文件首次不存在时的报错)；
  #   || true        ：grep 无匹配/出错返回非 0，强制成功，避免带偏脚本；
  #   ${before:-0}   ：为空(如文件缺失)时兜底为 0，保证是数字供算术比较。
  before="$(grep -c 'op: median=' "$RESULT_FILE" 2>/dev/null || true)"; before="${before:-0}"

  # 跑 benchmark：--result-file "$RESULT_FILE" 让 benchmark 把结构化结果【写到 autotune 要读的
  #   同一个文件】(否则 benchmark 写默认文件、本脚本 grep 的 $RESULT_FILE 永远没新行 → 误判无 OK)。
  #   2>&1 合并 stderr 到 stdout，| tee -a 让输出【同时】上终端(实时可见)并【追加】进 $OUTPUT 日志；
  #   "${BENCH_ARGS[@]}" 把参数数组逐元素安全展开。
  python "$SCRIPT_DIR/$BENCH_PY" "${BENCH_ARGS[@]}" --result-file "$RESULT_FILE" 2>&1 | tee -a "$OUTPUT"

  # after：跑之后再数一次。after>before ⟺ 本次新追加了 OK 行 ⟺ 本配置跑成功。
  after="$(grep -c 'op: median=' "$RESULT_FILE" 2>/dev/null || true)"; after="${after:-0}"

  if (( after > before )); then
    # 本次有新 OK：从 $RESULT_FILE 抠出【本次】op 的 perf 与 median。
    #   grep -oP        ：-o 只输出匹配片段，-P 用 Perl 正则(为了 \K)；
    #   '...perf=\K[0-9.]+'：前缀 'op: median=<数>ms perf=' 仅作定位，\K 把前缀丢出匹配，
    #                       故只输出 perf 后的数字；前缀锚定 op，故不会误抓 eager 的 perf；
    #   | tail -1       ：取最后一条匹配 = 本次追加的那行。
    op_gf="$(grep -oP 'op: median=[0-9.]+ms perf=\K[0-9.]+' "$RESULT_FILE" | tail -1)"
    op_ms="$(grep -oP 'op: median=\K[0-9.]+' "$RESULT_FILE" | tail -1)"   # 同理取 median 毫秒
    echo "RESULT: BLOCK_SIZE_X=$bsx  op median=${op_ms}ms  perf=${op_gf} GFLOPS" | tee -a "$OUTPUT"

    # 取最大 GFLOPS（浮点比较 bash (( )) 不支持，借 awk 的退出码当真假）：
    #   双引号让 $op_gf/$best_gf 先被替进 awk 程序文本，如 BEGIN{exit !(520.5 > 260.7)}；
    #   awk 内 (a>b) 真=1/假=0，!(...) 取反，exit N → 真时 exit 0；
    #   bash if 把 exit 0 当真 ⟹ op_gf>best_gf 时进 then，更新最佳。
    if awk "BEGIN{exit !($op_gf > $best_gf)}"; then best_gf="$op_gf"; best_bsx="$bsx"; fi
  else
    # after==before：本次没产出 OK(编译问题/SKIP/ERROR)，不去 tail 读旧值，只记一条说明。
    echo "RESULT: BLOCK_SIZE_X=$bsx  无 OK 输出（SKIP/ERROR，详见上方日志）" | tee -a "$OUTPUT"
  fi
done

# ── 总结：把 common.cuh 设回最佳值（若有），否则 trap 恢复原值 ──────────────────────
echo "" | tee -a "$OUTPUT"
echo "==================== AUTOTUNE 完成 ====================" | tee -a "$OUTPUT"
if [[ -n "$best_bsx" ]]; then
  echo "★最佳：BLOCK_SIZE_X=$best_bsx  →  ${best_gf} GFLOPS" | tee -a "$OUTPUT"
  trap - EXIT INT TERM                         # 取消恢复原值，改为写入最佳值
  sed -i "s/^#define BLOCK_SIZE_X .*/#define BLOCK_SIZE_X $best_bsx/" "$COMMON"
  echo "已把 common.cuh 的 BLOCK_SIZE_X 设为最佳值 $best_bsx（记得重编译 .so 生效）。" | tee -a "$OUTPUT"
else
  echo "没有任何配置产出有效性能（全部编译失败/SKIP）。common.cuh 已恢复原值 $ORIG_BSX。" | tee -a "$OUTPUT"
fi
echo "完整日志：$OUTPUT" | tee -a "$OUTPUT"
