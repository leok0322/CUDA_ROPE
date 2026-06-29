"""test_fused_qknorm_rope.py —— 多维度扫描，校验自定义融合算子是否正确。

复用 src/app/runner.py 的 Runner（eager 前向得参考 y_ref → torch.compile 把
"QK-Norm + RoPE" 子图替换成 torch.ops.ROPE_cuda.fused_qkv_norm_rope_{neox,interleave}
→ 比对编译后输出 y 与 y_ref），对【dtype × head_dim × interleave】各组合各跑一次，
逐组合打印 PASS / FAIL / SKIP，最后给出汇总表。

为什么【每个组合用独立子进程】：torch.compile / dynamo / PatternMatcher 在同一进程内跨
多组配置会有缓存与"替换 pass 重复注册"的状态串扰，子进程隔离最干净、结果最可信。

运行：
  ./run_check_op.sh                      # 经 venv，跑全量扫描
  uv run python src/scripts/test_fused_qknorm_rope.py            # 全量
  uv run python src/scripts/test_fused_qknorm_rope.py --dtype float16 --head-dim 128   # 过滤子集
  （内部用）... --worker --dtype float16 --head-dim 128 --interleave 1   # 单组合 worker
"""
import argparse
import os
import subprocess
import sys

# ── 让 import 找得到 src/app（util/runner/model…）与项目根（ROPE_cuda.so）─────────────
#   本文件在 src/scripts/fused_ROPE_RMSNorm/，上溯【3 级】到项目根，再把 src/app 与项目根加进 sys.path：
#     src/app   → util / runner / model / installer / rmsnorm_rope_replace_pass
#     项目根     → ROPE_cuda.*.so（CMake LIBRARY_OUTPUT_DIRECTORY=项目根）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
APP_DIR = os.path.join(PROJECT_ROOT, "src", "app")
for _p in (APP_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 扫描维度（默认全量；可经命令行 --dtype/--head-dim/--interleave/--token-sizes/--num-heads 过滤成子集）
ALL_DTYPES = ["float32", "float16", "bfloat16"]
ALL_HEAD_DIMS = [64, 128, 256]
ALL_INTERLEAVE = [False]
ALL_TOKEN_SIZES = [[4, 4], [3, 5, 2]]      # 各序列真实长度(ragged)：等长 / 变长各一
ALL_NUM_HEADS = [(8, 8, 8), (8, 2, 2)]     # (Hq,Hk,Hv)：MHA / GQA(Hk、Hv 小于 Hq) 各一


def _parse_int_list(s):
    """把逗号串解析成 int 列表，如 "8,2,2" -> [8,2,2]、"4,4" -> [4,4]。"""
    return [int(x) for x in str(s).split(",") if x != ""]

# allclose 容差按 dtype 放宽：低精度天然有更大的浮点偏差（融合 kernel vs eager 的累加顺序/精度）。
#   比较前统一 .float()，避免半精度自身 eps 触发误判。
TOL = {
    "float32": (1e-3, 1e-3),
    "float16": (1e-2, 1e-2),
    "bfloat16": (3e-2, 3e-2),
}

RESULT_PREFIX = "__RESULT__|"   # worker 打印的机器可解析结果行前缀


# ─────────────────────────────────────────────────────────────────────────────
# worker：在【独立子进程】里跑单个 (dtype, head_dim, interleave) 组合，打印一行结果。
# ─────────────────────────────────────────────────────────────────────────────
def run_worker(args):
    import torch
    import util
    import runner as runner_mod
    from runner import Runner

    def emit(status, detail=""):
        print(f"{RESULT_PREFIX}{status}|{detail}")

    dtype = getattr(torch, args.dtype)
    interleave = bool(args.interleave)
    rotary_dim = args.rotary_dim if args.rotary_dim and args.rotary_dim > 0 else None

    try:
        r = Runner(
            token_size=tuple(args.token_size), head_dim=args.head_dim, rotary_dim=rotary_dim,
            dtype=dtype, interleave=interleave, seed=args.seed,
            num_heads_q=args.num_heads_q, num_heads_k=args.num_heads_k, num_heads_v=args.num_heads_v,
        )
        x, positions = util.make_inputs_qkv(
            r.seed, r.num_heads_q, r.num_heads_k, r.num_heads_v,
            r.head_dim, r.device, r.dtype, r.token_size,
        )
        y_ref = r.run_eager(x, positions)
    except Exception as e:
        emit("ERROR", f"构造/eager 失败: {type(e).__name__}: {e}")
        return

    if not runner_mod._HAS_ROPE_CUDA:
        emit("SKIP", "ROPE_cuda 未加载（先用 CMake 构建 .so 并置于项目根）")
        return

    # 直接调 run_compiled（而非 run，run 会吞异常）以便把"运行期拒绝/算子未注册"区分出来。
    try:
        y = r.run_compiled(x, positions)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        low = str(e).lower()
        if any(k in str(e) for k in ("不支持当前 dtype", "packed_as", "BLOCK_SIZE_X", "rotary_dim")):
            emit("SKIP", f"运行期拒绝(算子守卫): {msg}")        # 预期内的非法组合/约束（如 float×256）
        elif any(k in low for k in ("no attribute", "no operator", "tried to", "could not find")):
            emit("SKIP", f"算子未注册: {msg}")                  # 如 neox 尚未在 interface.cpp 注册
        else:
            emit("ERROR", f"run_compiled 异常: {msg}")
        return

    if y is None:
        emit("SKIP", "无 compiled 输出（未发生融合替换）")
        return

    atol, rtol = TOL[args.dtype]
    same = torch.allclose(y_ref.float(), y.float(), atol=atol, rtol=rtol)
    if same:
        emit("PASS", f"allclose(atol={atol},rtol={rtol})")
    else:
        diff = (y_ref.float() - y.float()).abs()
        emit("FAIL", f"allclose=False  max|Δ|={diff.max().item():.3e} mean|Δ|={diff.mean().item():.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# parent：枚举组合 → 逐个起子进程跑 worker → 收集结果 → 打印汇总表 → 设退出码。
# ─────────────────────────────────────────────────────────────────────────────
def run_parent(args):
    dtypes = args.dtype or ALL_DTYPES
    head_dims = args.head_dim or ALL_HEAD_DIMS
    interleaves = ALL_INTERLEAVE if args.interleave is None else [bool(args.interleave)]
    token_sizes = [_parse_int_list(s) for s in args.token_sizes] if args.token_sizes else ALL_TOKEN_SIZES
    num_heads = [tuple(_parse_int_list(s)) for s in args.num_heads] if args.num_heads else ALL_NUM_HEADS

    combos = [(dt, hd, il, ts, nh)
              for dt in dtypes for hd in head_dims for il in interleaves
              for ts in token_sizes for nh in num_heads]
    print(f"[sweep] 共 {len(combos)} 个组合："
          f"dtype×head_dim×interleave×token_size×num_heads = "
          f"{dtypes} × {head_dims} × {[int(i) for i in interleaves]} × {token_sizes} × {num_heads}\n")

    rows = []
    for dt, hd, il, ts, nh in combos:
        hq, hk, hv = nh
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--dtype", dt, "--head-dim", str(hd), "--interleave", str(int(il)),
            "--token-size", *[str(t) for t in ts],
            "--num-heads-q", str(hq), "--num-heads-k", str(hk), "--num-heads-v", str(hv),
            "--seed", str(args.seed),
        ]
        if args.rotary_dim:
            cmd += ["--rotary-dim", str(args.rotary_dim)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        status, detail = "ERROR", "worker 未输出结果行"
        for line in proc.stdout.splitlines():
            if line.startswith(RESULT_PREFIX):
                _, status, detail = line.split("|", 2)
        if status == "ERROR" and detail == "worker 未输出结果行":
            # worker 崩溃：附上 stderr 末尾几行帮助定位
            tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
            detail = f"worker 崩溃(rc={proc.returncode})  stderr尾: {tail}"

        tag = {"PASS": "✅PASS", "FAIL": "❌FAIL", "SKIP": "⏭ SKIP", "ERROR": "💥ERROR"}.get(status, status)
        label = (f"{dt:9s} hd={hd:<3d} il={int(il)} "
                 f"ts={','.join(map(str, ts)):9s} heads={hq}/{hk}/{hv}")
        print(f"  {tag}  {label}  | {detail}")
        rows.append((status, label, detail))

    # 汇总
    n_pass = sum(1 for s, *_ in rows if s == "PASS")
    n_fail = sum(1 for s, *_ in rows if s == "FAIL")
    n_skip = sum(1 for s, *_ in rows if s == "SKIP")
    n_err = sum(1 for s, *_ in rows if s == "ERROR")
    print("\n" + "=" * 72)
    print(f"汇总：PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  ERROR={n_err}  (共 {len(rows)})")
    if n_fail or n_err:
        print("结果：❌ 有 FAIL/ERROR —— 自定义算子在某些组合上不正确或出错（见上表）。")
    else:
        print("结果：✅ 所有【实际产生输出】的组合 allclose 通过；SKIP 为未注册/受约束跳过，非错误。")
    print("=" * 72)
    # 退出码：只有真正的 FAIL/ERROR 才算套件失败；SKIP（受约束/未注册）不算。
    return 1 if (n_fail or n_err) else 0


def build_parser():
    p = argparse.ArgumentParser(description="多维度扫描校验 fused QK-Norm + RoPE 自定义算子")
    p.add_argument("--worker", action="store_true", help="（内部）单组合 worker 模式")
    p.add_argument("--dtype", nargs="*", choices=ALL_DTYPES,
                   help="只测这些 dtype（worker 模式下为单值）；缺省=全部")
    p.add_argument("--head-dim", dest="head_dim", nargs="*", type=int,
                   help="只测这些 head_dim；缺省=64/128/256")
    p.add_argument("--interleave", type=int, choices=[0, 1], default=None,
                   help="只测某个 RoPE 风格(0=neox,1=interleave)；缺省=两者都测")
    p.add_argument("--rotary-dim", dest="rotary_dim", type=int, default=0,
                   help="rotary_dim；0/缺省=全旋转(=head_dim)")
    # ── parent 扫描维度：token_size 与 num_heads 也作为被扫的维度 ──
    p.add_argument("--token-sizes", dest="token_sizes", nargs="*", default=None,
                   help='扫描的 token_size 集合，每项逗号串，如 --token-sizes 4,4 3,5,2；缺省=内置集')
    p.add_argument("--num-heads", dest="num_heads", nargs="*", default=None,
                   help='扫描的 (Hq,Hk,Hv) 集合，每项逗号串，如 --num-heads 8,8,8 8,2,2；缺省=内置集')
    # ── worker 单组合参数（由 parent 逐组合传入；命令行一般不直接用）──
    p.add_argument("--token-size", dest="token_size", nargs="*", type=int, default=[4, 4],
                   help="（worker）单个 token_size，各序列真实长度列表(ragged)")
    p.add_argument("--num-heads-q", dest="num_heads_q", type=int, default=8, help="（worker）单个 Hq")
    p.add_argument("--num-heads-k", dest="num_heads_k", type=int, default=8, help="（worker）单个 Hk")
    p.add_argument("--num-heads-v", dest="num_heads_v", type=int, default=8, help="（worker）单个 Hv")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.worker:
        # worker 模式：--dtype/--head-dim 视为单值（取列表首元素）
        args.dtype = (args.dtype or ["float32"])[0]
        args.head_dim = (args.head_dim or [128])[0]
        args.interleave = args.interleave if args.interleave is not None else 0
        run_worker(args)
        sys.exit(0)
    else:
        sys.exit(run_parent(args))
