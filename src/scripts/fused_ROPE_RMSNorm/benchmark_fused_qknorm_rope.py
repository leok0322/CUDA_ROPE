"""benchmark_fused_qknorm_rope.py —— 自定义融合算子 vs eager 的耗时对比（多配置扫描）。

参考 LayerNorm 的 benchmark 脚本（CUDA Event 计时 + 结果追加写文件），但本算子的
"自定义算子"路径通过 torch.compile 把模型里的 "QK-Norm + RoPE" 子图替换成
torch.ops.ROPE_cuda.fused_qkv_norm_rope_{neox,interleave}，故对比的是：
  · eager    = FusedQKVNormRope.forward（未融合的分步子图）
  · 自定义算子 = torch.compile(model) 的热路径（子图已被替换成一次融合 kernel）
两者各跑 WARMUP 预热 + REPEATS 次 CUDA Event 计时，并先用 allclose 校验数值一致。

每个配置在【独立子进程】里跑：torch.compile/dynamo/PatternMatcher 跨配置有状态串扰，
子进程隔离最干净（与 test_fused_qknorm_rope.py 同思路）。

── 使用的数据 & 维度（简要）────────────────────────────────────────────────────
  · 数据：util.make_inputs_qkv 随机生成（torch.randn）——
      x        : [num_tokens, (Hq+Hk+Hv)*head_dim]，每 token 内 [Q…|K…|V…] 连续
      positions: [num_tokens]，各序列按 token_size 分段 arange（ragged）
      γ / cos / sin 来自 Runner.build_model() 随机初始化的模型；num_tokens = sum(token_size)
  · 扫描维度（默认值见 BENCH_*，默认在 dtype × head_dim × token_size 三维铺开）：
      dtype        BENCH_DTYPES      = [float16, bfloat16, float32]
      head_dim     BENCH_HEAD_DIMS   = [64, 128, 256]
      interleave   BENCH_INTERLEAVE  = [False(neox)]
      token_size   BENCH_TOKEN_SIZES = [[128],[512],[2048],[8192]] → num_tokens=128/512/2048/8192（规模轴）
      num_heads    BENCH_NUM_HEADS   = [(8,8,8)]  即 (Hq,Hk,Hv)
    默认 3×3×1×4×1 = 36 个配置；非法组合(如 float32×256)由算子守卫优雅 SKIP，不计失败。
    基准(baseline)=eager 子图；自定义算子=torch.compile 融合 kernel；speedup=eager_median/op_median。
────────────────────────────────────────────────────────────────────────────────

运行：
  ./run_benchmark_op.sh                                   # 经 venv，跑默认扫描
  uv run python src/scripts/benchmark_fused_qknorm_rope.py --token-sizes 256 1024 4096
  （内部用）... --worker --dtype float16 --head-dim 128 --interleave 0 --token-size 512 ...
"""
import argparse
import datetime
import os
import subprocess
import sys

# ── 让 import 找得到 src/app（util/runner/...）、项目根与 cmake-build*（ROPE_cuda.so）────
#   本文件在 src/scripts/fused_ROPE_RMSNorm/，上溯【3 级】到项目根。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
APP_DIR = os.path.join(PROJECT_ROOT, "src", "app")
_search = [APP_DIR, PROJECT_ROOT]
for _b in ("cmake-build-release", "cmake-build-debug"):   # .so 可能在构建目录而非项目根
    _c = os.path.join(PROJECT_ROOT, _b)
    if os.path.isdir(_c):
        _search.append(_c)
for _p in _search:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 计时参数（与参考脚本一致）
REPEATS = 100
WARMUP = 10

# 默认扫描维度（benchmark 较慢；可经命令行进一步过滤/扩展）。
#   现默认在 dtype × head_dim × token_size 三维上铺开（interleave、num_heads 仍单值）：
#     笛卡尔积 = 3×3×1×4×1 = 36 个配置（每配置一个子进程 + torch.compile，整体较慢）。
#     部分非法组合会被算子守卫优雅 SKIP，不算失败：
#       · float32 × head_dim=256 → 每线程 8 个 4 字节 → packed_as<float,8> 不存在 → SKIP；
#       · head_dim=64 需 .so 已去掉 half%BLOCK_SIZE_X 检查（否则 half=32 不整除 BLOCK_SIZE_X 时 SKIP）。
#   想收窄：经 CLI 传单值，如 --dtype float16 --head-dim 128 --token-sizes 2048
BENCH_DTYPES = ["float16", "bfloat16", "float32"]   # 多值→变；half/bf16 全 head_dim 合法，float×256 非法(SKIP)
BENCH_HEAD_DIMS = [64, 128, 256]                    # 多值→变
BENCH_INTERLEAVE = [False]                      # 单值→不变；与默认 config 一致(neox)，按已注册算子调整
BENCH_TOKEN_SIZES = [[128], [512], [2048], [8192]]   # 多值→变；num_tokens = sum(token_size)，规模轴
BENCH_NUM_HEADS = [(8, 8, 8)]                   # 单值→不变；(Hq,Hk,Hv)

RESULT_PREFIX = "__RESULT__|"
RESULT_DIR = os.path.join(PROJECT_ROOT, "benchmark_results")
# 结果文件【默认】路径；可经 --result-file 覆盖（如让 autotune 指到它要读的那个文件）。
DEFAULT_RESULT_FILE = os.path.join(RESULT_DIR, "ROPE_python_op_benchmark_result.txt")


def _parse_int_list(s):
    return [int(x) for x in str(s).split(",") if x != ""]


# ─────────────────────────────────────────────────────────────────────────────
# worker：单配置子进程——建模型/编译 → 校验一致 → CUDA Event 计时 eager 与 自定义算子。
# ─────────────────────────────────────────────────────────────────────────────
def run_worker(args):
    import torch
    import util
    import runner as runner_mod
    from runner import Runner
    # ── 两种导入方式，二选一成功即可（兼容不同启动方式）──────────────────────────
    #
    # 【脚本加载(裸名)】 from rmsnorm_rope_replace_pass import ...
    #   把模块当【顶层模块】导入(名字里无包前缀)。
    #   前提：模块【所在目录 src/app】本身在 sys.path 上 → 该 .py 作为顶层模块可直接找到。
    #   本文件顶部 sys.path.insert(0, APP_DIR) 已把 src/app 加进来，故正常走这条(try)。
    #
    # 【包加载(带点路径)】 from src.app.rmsnorm_rope_replace_pass import ...
    #   通过【包路径 src.app.X】从项目根逐级进包寻址。
    #   前提：项目根在 sys.path 上，且 src、src.app 能作为包被解析(常规包或命名空间包均可)。
    #   本文件顶部 sys.path.insert(0, PROJECT_ROOT) 亦满足，故作 except 兜底。
    #
    # 关键区别只在【名字带不带包前缀】：裸名=顶层模块(靠模块目录在 path)；
    #   src.app.X=包成员(靠项目根在 path，逐级把目录当包)。两条都是【绝对导入】。
    #   注：本文件是被当【脚本】跑(python …/benchmark.py)，__name__=="__main__" 无父包，
    #       故用不了 runner.py 的 `from . import`(相对导入)，只能用上面这对绝对导入。
    #
    # ── 常规包 vs 命名空间包(决定上面"能作为包被解析"的含义)──────────────────────
    #   · 常规包(regular package)   ：目录【有 __init__.py】。导入时执行其中代码(可做初始化/重导出)。
    #                                 本项目 src/app/__init__.py 存在 → src.app 是常规包。
    #   · 命名空间包(namespace pkg) ：目录【没有 __init__.py】，PEP 420(Py3.3+)起仍可当包，
    #                                 只要其父目录在 sys.path 上即可逐级解析子模块；无初始化代码，
    #                                 __path__ 可跨多个目录。本项目 src 无 __init__.py → 是命名空间包。
    #   ⟹ "有没有 __init__.py" 只决定【是常规包还是命名空间包 / 能否放初始化代码】，
    #      【不】决定"能否被解析到"——两类包都能有子模块、都能被 import。
    try:
        from rmsnorm_rope_replace_pass import RMSNormRoPEreplacePass
        from installer import Installer
    except ImportError:  # 兼容包/脚本两种加载方式
        from src.app.rmsnorm_rope_replace_pass import RMSNormRoPEreplacePass  # noqa
        from src.app.installer import Installer  # noqa

    def emit(status, num_tokens="-", op_med="-", eager_med="-", speedup="-",
             op_gflops="-", eager_gflops="-", correct="-", detail=""):
        print(f"{RESULT_PREFIX}{status}|{num_tokens}|{op_med}|{eager_med}|{speedup}"
              f"|{op_gflops}|{eager_gflops}|{correct}|{detail}")

    # ── CUDA Event 计时（精度 ~0.5us），返回每次的 ms 列表 ──
    def benchmark(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEATS)]
        for i in range(REPEATS):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]

    dtype = getattr(torch, args.dtype)
    interleave = bool(args.interleave)
    rotary_dim = args.rotary_dim if args.rotary_dim and args.rotary_dim > 0 else None

    # ---- 建模型与输入 ----
    # max_token_size = cos/sin cache 行数上限，须 >= 最长序列(max(token_size))，否则 positions 越界。
    #   --max-token-size>0 用指定值；否则自动取 max(4096, 最长序列)，保证大 token_size 也能跑。
    max_token_size = args.max_token_size if args.max_token_size > 0 else max(4096, max(args.token_size))
    try:
        r = Runner(
            token_size=tuple(args.token_size), head_dim=args.head_dim, rotary_dim=rotary_dim,
            max_token_size=max_token_size,
            dtype=dtype, interleave=interleave, seed=args.seed,
            num_heads_q=args.num_heads_q, num_heads_k=args.num_heads_k, num_heads_v=args.num_heads_v,
        )
        r.build_model()
        x, positions = util.make_inputs_qkv(
            r.seed, r.num_heads_q, r.num_heads_k, r.num_heads_v,
            r.head_dim, r.device, r.dtype, r.token_size,
        )
    except Exception as e:
        emit("ERROR", detail=f"构造失败: {type(e).__name__}: {e}")
        return

    if r.device != "cuda" or not torch.cuda.is_available():
        emit("SKIP", num_tokens=r.num_tokens, detail="无 CUDA 设备，benchmark 需 GPU")
        return
    if not runner_mod._HAS_ROPE_CUDA:
        emit("SKIP", num_tokens=r.num_tokens, detail="ROPE_cuda 未加载（先构建 .so）")
        return

    # ---- 安装融合 pass + torch.compile（复刻 Runner.run_compiled 的装配，保留 compiled 以便重复计时）----
    try:
        op_name = ("ROPE_cuda::fused_qkv_norm_rope_interleave" if interleave
                   else "ROPE_cuda::fused_qkv_norm_rope_neox")
        rp = RMSNormRoPEreplacePass(
            eps=r.eps, num_heads_q=r.num_heads_q, num_heads_k=r.num_heads_k,
            num_heads_v=r.num_heads_v, head_dim=r.head_dim, rotary_dim=r.rotary_dim,
            op_name=op_name, interleave=interleave,
        )
        example_inputs = rp.make_example_inputs(r.num_tokens, r.device, r.dtype)
        installer = Installer(rp.search_fn, rp.replace_fn, example_inputs, rp.op_name, rp.make_fake_fn())
        installer.install_fusion_pass()
        compiled = torch.compile(r.model)
    except Exception as e:
        emit("ERROR", num_tokens=r.num_tokens, detail=f"装配/compile 失败: {type(e).__name__}: {e}")
        return

    # ---- 正确性：各用一份干净拷贝（融合算子对 qkv 就地改写，故必须 clone）----
    try:
        with torch.no_grad():
            y_ref = r.model(x.clone(), positions)
            y = compiled(x.clone(), positions)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        low = str(e).lower()
        if any(k in str(e) for k in ("不支持当前 dtype", "packed_as", "rotary_dim")):
            emit("SKIP", num_tokens=r.num_tokens, detail=f"运行期拒绝(算子守卫): {msg}")
        elif any(k in low for k in ("no attribute", "no operator", "tried to", "could not find")):
            emit("SKIP", num_tokens=r.num_tokens, detail=f"算子未注册: {msg}")
        else:
            emit("ERROR", num_tokens=r.num_tokens, detail=f"前向异常: {msg}")
        return

    if y is None:
        emit("SKIP", num_tokens=r.num_tokens, detail="无 compiled 输出（未发生融合替换）")
        return
    correct = bool(torch.allclose(y_ref.float(), y.float(), atol=3e-2, rtol=3e-2))

    # ---- 计时：eager 与 自定义算子（用各自的 buffer；就地改写只影响数值不影响耗时）----
    xe = x.clone()
    xo = x.clone()
    eager_fn = lambda: r.model(xe, positions)              # noqa: E731
    op_fn = lambda: compiled(xo, positions)               # noqa: E731
    with torch.no_grad():
        t_op = benchmark(op_fn)
        t_eager = benchmark(eager_fn)

    op_med = sorted(t_op)[len(t_op) // 2]        # ms
    eager_med = sorted(t_eager)[len(t_eager) // 2]
    speedup = eager_med / op_med if op_med > 0 else float("nan")

    # ── FLOP 计数（仅算术运算，与参考 C++ 一致：sin/cos/rsqrt 等超越函数不计）──────────
    #   仅 Q/K 头参与计算（V 透传，0 FLOP），逐 token 逐 head：
    #     RMSNorm  ：每元素 ≈ 4 FLOP（x² 1 mul + 累加 1 add + 归一化 1 mul + 乘权重 γ 1 mul）
    #                → 每 head 4*head_dim
    #     RoPE     ：每"维度对"(2 元素) 6 FLOP（x1*cos - x2*sin / x1*sin + x2*cos：各 2mul+1add）
    #                → 每 head 有 rotary_dim/2 个维度对 → 6*(rotary_dim/2) = 3*rotary_dim
    #   每 head 小计 = 4*head_dim + 3*rotary_dim；× (Hq+Hk) 个 QK 头 × num_tokens 行
    rd = r.rotary_dim
    flops = r.num_tokens * (r.num_heads_q + r.num_heads_k) * (4 * r.head_dim + 3 * rd)
    op_gflops = flops / (op_med / 1e3) / 1e9 if op_med > 0 else float("nan")
    eager_gflops = flops / (eager_med / 1e3) / 1e9 if eager_med > 0 else float("nan")

    emit("OK", num_tokens=r.num_tokens,
         op_med=f"{op_med:.4f}", eager_med=f"{eager_med:.4f}", speedup=f"{speedup:.2f}",
         op_gflops=f"{op_gflops:.1f}", eager_gflops=f"{eager_gflops:.1f}",
         correct=("PASS" if correct else "FAIL"))


# ─────────────────────────────────────────────────────────────────────────────
# parent：枚举配置 → 逐个子进程跑 worker → 打印表 → 追加写 benchmark_results。
# ─────────────────────────────────────────────────────────────────────────────
def run_parent(args):
    dtypes = args.dtype or BENCH_DTYPES
    head_dims = args.head_dim or BENCH_HEAD_DIMS
    interleaves = BENCH_INTERLEAVE if args.interleave is None else [bool(args.interleave)]
    token_sizes = [_parse_int_list(s) for s in args.token_sizes] if args.token_sizes else BENCH_TOKEN_SIZES
    num_heads = [tuple(_parse_int_list(s)) for s in args.num_heads] if args.num_heads else BENCH_NUM_HEADS

    combos = [(dt, hd, il, ts, nh)
              for dt in dtypes for hd in head_dims for il in interleaves
              for ts in token_sizes for nh in num_heads]
    print(f"[bench] 共 {len(combos)} 个配置；REPEATS={REPEATS} WARMUP={WARMUP}\n")
    header = (f"{'config':<46}  {'num_tok':>7}  {'op median':>10}  {'op GF/s':>9}  "
              f"{'eager median':>12}  {'eager GF/s':>10}  {'eager/op':>8}  {'correct':>7}")
    print(header)
    print("-" * len(header))

    rows = []
    for dt, hd, il, ts, nh in combos:
        hq, hk, hv = nh
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--dtype", dt, "--head-dim", str(hd), "--interleave", str(int(il)),
            "--token-size", *[str(t) for t in ts],
            "--num-heads-q", str(hq), "--num-heads-k", str(hk), "--num-heads-v", str(hv),
            "--max-token-size", str(args.max_token_size),
            "--seed", str(args.seed),
        ]
        if args.rotary_dim:
            cmd += ["--rotary-dim", str(args.rotary_dim)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        parts = None
        for line in proc.stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                parts = line[len(RESULT_PREFIX):].split("|", 8)
        if parts is None:
            tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
            parts = ["ERROR", "-", "-", "-", "-", "-", "-", "-",
                     f"worker 崩溃(rc={proc.returncode}) {tail}"]
        status, num_tok, op_med, eager_med, speedup, op_gf, eager_gf, correct, detail = parts

        cfg = f"{dt:9s} hd={hd:<3d} il={int(il)} ts={','.join(map(str, ts)):9s} heads={hq}/{hk}/{hv}"
        if status == "OK":
            print(f"{cfg:<46}  {num_tok:>7}  {op_med+'ms':>10}  {op_gf:>9}  "
                  f"{eager_med+'ms':>12}  {eager_gf:>10}  {speedup+'x':>8}  {correct:>7}")
        else:
            print(f"{cfg:<46}  {num_tok:>7}  {('⏭ '+status):>10}  | {detail}")
        rows.append((cfg, status, num_tok, op_med, eager_med, speedup, op_gf, eager_gf, correct, detail))

    _append_results(rows, dtypes, head_dims, interleaves, token_sizes, num_heads, args.result_file)
    n_ok = sum(1 for r in rows if r[1] == "OK")
    n_fail = sum(1 for r in rows if r[1] == "OK" and r[8] == "FAIL")
    n_err = sum(1 for r in rows if r[1] == "ERROR")
    print(f"\n汇总：OK={n_ok}（其中数值 FAIL={n_fail}）  ERROR={n_err}  其余 SKIP；详见上表")
    print(f"结果已追加：{args.result_file}")
    return 1 if n_err else 0


def _append_results(rows, dtypes, head_dims, interleaves, token_sizes, num_heads, result_file):
    # result_file 可经 --result-file 指定；其所在目录自动创建（dirname 为空时落当前目录）。
    os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n[{ts}]  repeats={REPEATS} warmup={WARMUP}  "
             f"sweep: dtype={dtypes} head_dim={head_dims} interleave={[int(i) for i in interleaves]} "
             f"token_sizes={token_sizes} num_heads={num_heads}"]
    for cfg, status, num_tok, op_med, eager_med, speedup, op_gf, eager_gf, correct, detail in rows:
        if status == "OK":
            lines.append(f"  {cfg}  num_tok={num_tok}  "
                         f"op: median={op_med}ms perf={op_gf} GFLOPS  "
                         f"eager: median={eager_med}ms perf={eager_gf} GFLOPS  "
                         f"eager/op={speedup}x  correct={correct}")
        else:
            lines.append(f"  {cfg}  {status}: {detail}")
    with open(result_file, "a") as f:
        f.write("\n".join(lines) + "\n")


def build_parser():
    p = argparse.ArgumentParser(description="自定义融合算子 vs eager 耗时对比（多配置扫描）")
    p.add_argument("--worker", action="store_true", help="（内部）单配置 worker 模式")
    p.add_argument("--dtype", nargs="*", choices=["float32", "float16", "bfloat16"],
                   help="只测这些 dtype（worker 模式下为单值）；缺省=内置集")
    p.add_argument("--head-dim", dest="head_dim", nargs="*", type=int, help="只测这些 head_dim")
    p.add_argument("--interleave", type=int, choices=[0, 1], default=None,
                   help="RoPE 风格(0=neox,1=interleave)；缺省=内置集")
    p.add_argument("--rotary-dim", dest="rotary_dim", type=int, default=0, help="0=全旋转(=head_dim)")
    p.add_argument("--token-sizes", dest="token_sizes", nargs="*", default=None,
                   help='扫描的 token_size 集合，每项逗号串，如 --token-sizes 512 2048 4,4')
    p.add_argument("--num-heads", dest="num_heads", nargs="*", default=None,
                   help='扫描的 (Hq,Hk,Hv) 集合，每项逗号串，如 --num-heads 8,8,8 32,8,8')
    p.add_argument("--result-file", dest="result_file", default=DEFAULT_RESULT_FILE,
                   help=f"结果追加写入的文件路径（缺省={DEFAULT_RESULT_FILE}）")
    # worker 单配置参数（parent 逐配置传入）
    p.add_argument("--token-size", dest="token_size", nargs="*", type=int, default=[512])
    p.add_argument("--num-heads-q", dest="num_heads_q", type=int, default=8)
    p.add_argument("--num-heads-k", dest="num_heads_k", type=int, default=8)
    p.add_argument("--num-heads-v", dest="num_heads_v", type=int, default=8)
    p.add_argument("--max-token-size", dest="max_token_size", type=int, default=0,
                   help="cos/sin cache 行数上限；0=自动=max(4096, 最长序列)，须 >= max(token_size)")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.worker:
        args.dtype = (args.dtype or ["float16"])[0]
        args.head_dim = (args.head_dim or [128])[0]
        args.interleave = args.interleave if args.interleave is not None else 0
        run_worker(args)
        sys.exit(0)
    else:
        sys.exit(run_parent(args))
