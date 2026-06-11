import os

import torch
import yaml


def _build_dtype_map(raw_map):
    """把 YAML 里的 dtype_map（别名 -> torch dtype 名字符串）解析成 别名 -> torch.dtype。

    YAML 存不了 torch.float32 这类对象，故配置里存的是属性名字符串（如 "float32"），
    这里逐项 getattr(torch, 名) 还原成真正的 torch.dtype；键统一小写，便于大小写不敏感查表。
    """
    return {str(k).lower(): getattr(torch, str(v)) for k, v in raw_map.items()}


def load_config(path=None):
    """读取 YAML 配置文件，返回可直接 Runner(**cfg) 展开的配置字典。

    path 缺省为本文件同目录下的 config/config.yaml。dtype 字段是字符串，经 YAML 里的
    dtype_map（_build_dtype_map 解析）映射到 torch.dtype；device 为 null 时保留 None，
    交给 Runner.__init__ 自动选。
    放在 util(而非 Runner.classmethod)里：避免 util 反向 import Runner 造成循环导入，
    故只负责"解析配置"，构造 Runner 留给调用方。
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # YAML 分两个顶层分组(run_params / model_arch)，各自展平合并成一个扁平 dict，
    # 使 Runner(**cfg) 仍能按形参一一对应地直接展开。
    run_params = raw.get("run_params", {}) or {}
    model_arch = raw.get("model_arch", {}) or {}
    cfg = {**run_params, **model_arch}
    # 仅当对应键存在时才做类型转换（缺省则交给 Runner.__init__ 的默认值）
    # dtype 字符串 → torch.dtype：映射表 dtype_map 也来自 YAML（顶层键），不再写死在代码里
    if "dtype" in cfg:
        dtype_map = _build_dtype_map(raw.get("dtype_map", {}) or {})
        cfg["dtype"] = dtype_map[str(cfg["dtype"]).lower()]
    # eps/base 显式转 float（防止 YAML 把它们写成字符串时出错）
    if "eps" in cfg:
        cfg["eps"] = float(cfg["eps"])
    if "base" in cfg:
        cfg["base"] = float(cfg["base"])
    return cfg


def build_cos_sin_cache(max_position, rotary_dim, base=10000.0,
                        dtype=torch.float32, device="cpu"):
    """构造 [max_position, rotary_dim] 的 RoPE cos/sin cache（前 cos 后 sin）。

    cache 布局与 CUDA kernel (src/inference/fused_qknorm_rope_kernel.cu) 保持一致：
      形状 [max_position, rotary_dim]，每行【前半段是 cos、后半段是 sin】，
      每段长度 half = rotary_dim // 2。第 pos 行、第 i 个频率：
          inv_freq[i] = 1 / base ** (2i / rotary_dim)      i = 0 .. half-1
          angle       = pos * inv_freq[i]
          cos_part[pos, i] = cos(angle)；sin_part[pos, i] = sin(angle)
    同一份 cache 既能喂给 PyTorch 参考实现 (model.py)，也能喂给 CUDA kernel。

    参数：
      max_position : 支持的最大位置数（cache 行数）
      rotary_dim   : RoPE 作用的维度数（必须为偶数）
      base         : 频率底数，常用 10000
      dtype/device : 输出张量的精度与设备
    返回：
      cache: torch.Tensor，形状 [max_position, rotary_dim]
             cache[:, :half] = cos，cache[:, half:] = sin
    """
    assert rotary_dim % 2 == 0, "rotary_dim 必须为偶数"
    # ★ 中间全程用 float32 计算，最后一步才 cast 到目标 dtype（return 的 cache.to(dtype)）。
    #   不能用参数 dtype 算 arange/幂/三角：低精度会坏精度甚至坏正确性——
    #   - positions=arange(max_position)：fp16 连续整数仅精确到 2048、bf16 仅到 256，
    #     max_position 一大(如 4096)，position 被量化、角度 pos*inv_freq 全错(eager 也错)；
    #   - inv_freq 跨 ~4 个数量级、cos/sin 角度大，低精度有效位不够，误差被放大。
    #   这也是 HF/vLLM 的标准做法：cache 在 fp32 上构造，仅最终结果转模型精度。
    # 频率 inv_freq[i] = 1 / base^(2i/rotary_dim)，长度 half = rotary_dim/2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                 / rotary_dim)
    )                                                       # [half]
    positions = torch.arange(max_position, dtype=torch.float32, device=device)  # [max_position]
    freqs = torch.outer(positions, inv_freq)                # [max_position, half] = pos*inv_freq
    cos = torch.cos(freqs)                                  # [max_position, half]
    sin = torch.sin(freqs)                                  # [max_position, half]
    cache = torch.cat([cos, sin], dim=-1)                   # [max_position, rotary_dim] 前 cos 后 sin
    return cache.to(dtype)


def make_inputs_qkv(seed, num_heads_q, num_heads_k, num_heads_v, head_dim,
                    device, dtype, token_size):
    """合并 qkv 输入: [num_tokens, (Hq+Hk+Hv)*head_dim]，及每 token 的 position。
    每 token 内按 [Q 头…|K 头…|V 头…] 连续排布(与 FusedQKVNormRope / 真实 kernel 一致)。

    token_size: 每条序列的【真实 token 数】列表(ragged，变长)。
      num_tokens = sum(token_size)，逐条真实相加、无 padding(取代旧的 batch*token_size)。
      positions = 把各序列各自的 arange(len) 按顺序拼接 → [num_tokens]，
                  例 token_size=[3,5] → positions=[0,1,2, 0,1,2,3,4]。"""
    torch.manual_seed(seed)
    total = num_heads_q + num_heads_k + num_heads_v
    num_tokens = sum(token_size)
    qkv = torch.randn(num_tokens, total * head_dim, device=device, dtype=dtype)
    # 变长(ragged)：每条序列各自 0..len-1，按序列顺序拼接成 [num_tokens]，无 padding
    positions = torch.cat([torch.arange(n, device=device) for n in token_size])
    return qkv, positions



def _rotate_half(x):
    """neox 风格旋转辅助：把最后一维分成前后两半 [x1, x2] -> [-x2, x1]。"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

# ─────────────────────────────────────────────────────────────────────────────
# 纯函数（不依赖 self/buffer），供【模型 forward】与【torch.compile 的 search_fn】
# 共用 —— 即“把核心数学抽成纯函数、两边共用”，避免重复实现导致漂移。
# 约定张量形状：x [num_tokens, num_heads, head_dim]；cos/sin [num_tokens, half]。
# ─────────────────────────────────────────────────────────────────────────────
def rms_norm_pure(x, weight, eps):
    # ---- 步骤 1：最后一维 RMSNorm ----
    """最后一维 RMSNorm：在 float32 上算 mean(x^2)，再 * 可学习 γ(weight)。"""
    in_dtype = x.dtype
    xf = x.float()
    # 均方值 ms = mean(x^2)（RMSNorm 的核心统计量，即二阶原点矩 E[X^2]）：
    #   x.pow(2)         逐元素平方，形状不变 [.., head_dim]
    #   .mean(dim=-1)    沿最后一维(特征维 head_dim)求平均，对每个 (token,head) 行
    #                    取该行 head_dim 个平方值的均值；【不减均值】(区别于 LayerNorm)
    #   keepdim=True     保留末维为 1 → [.., 1]，便于下一步与 x 广播相除
    ms = xf.pow(2).mean(dim=-1, keepdim=True)        # 二阶原点矩 mean(x^2)
    # x/[..,head_dim] 乘 rsqrt(ms)/[..,1]：靠广播自动把末维 1 拉伸到 head_dim，
    #   即每行 head_dim 个特征同乘一个标量(该行 RMS 的倒数)。依赖上面 keepdim=True
    #   保留末维为 1；否则形状 [..,head_dim]*[..] 从右对齐失败需手动 unsqueeze(-1)。
    #   广播是逻辑扩展、不复制数据。
    xf = xf * torch.rsqrt(ms + eps)                  # x / sqrt(mean(x^2)+eps)
    return xf.to(in_dtype) * weight                  # 逐维乘 γ


def rope_pure(x, cos, sin, interleave=False):
     # ---- 步骤 2：最后一维 RoPE ----
    """对 x 的最后一维做 RoPE(此处 rotary_dim==head_dim，整段都旋转)。
    cos/sin: [num_tokens, half]，在 head 维广播(同 token 各 head 共享角度)。"""
    # 索引里 None = 在该位置插一个 size=1 的新维(等价 unsqueeze(1))，: 表示该维原样保留。
    #   cos: [num_tokens, half] -> [num_tokens, 1, half]，中间补出的就是 head 维。
    #   cos/sin 本无 head 维：RoPE 旋转角只与 position 和频率 i 有关、与是哪个 head 无关，
    #   故同一 token 的所有 head 共享同一组角度。补出长度 1 后，与 x[num_tokens,num_heads,..]
    #   相乘时这个 1 沿 head 维广播成 num_heads，即每个 head 都乘同一份 cos/sin。
    c = cos[:, None, :]                              # [num_tokens, 1, half]
    s = sin[:, None, :]
    if interleave:
        # interleave(=!neox)：相邻 (2i, 2i+1) 配对
        # 切片 [..., start::step]：... 保留前面所有维、只切最后一维(rotary_dim)；
        #   0::2 = 从下标0起步长2 → 偶数位 0,2,4,…；1::2 = 从1起步长2 → 奇数位 1,3,5,…
        #   即把相邻一对 (2i,2i+1) 分别抽进 x1[i]=rot[2i]、x2[i]=rot[2i+1]，各 half 长。
        x1 = x[..., 0::2]                            # 偶数位 -> [.., half]
        x2 = x[..., 1::2]                            # 奇数位 -> [.., half]
        # 每对 (x1,x2) 用同一角度 θ 旋转：r1=x1cosθ-x2sinθ, r2=x1sinθ+x2cosθ
        # cos/sin 是 [.,1,half]，与 x1[.,num_heads,half] 相乘时中间 1 广播成 num_heads，
        # 故 r1、r2 形状均为 [num_tokens, num_heads, half]。
        r1 = x1 * c - x2 * s
        r2 = x1 * s + x2 * c
        # torch.stack：在【新建维度】上叠同形状张量(区别于 cat 沿已有维拼接，不增维)。
        #   dim=-1 把长度2的新维插到最后：[.,num_heads,half] -> [.,num_heads,half,2]，
        #   末维 [r1[i], r2[i]] 即每个频率位的一对。
        # flatten(-2)：合并倒数两维 half*2=rotary_dim，按最内先变展开成
        #   r1[0],r2[0],r1[1],r2[1],… 的【交错】排布(若用 cat 会得到前半全r1/后半全r2，
        #   那是 neox 排布，不是 interleave)。
        return torch.stack([r1, r2], dim=-1).flatten(-2)
    # neox：第 j 维 ↔ 第 j+half 维(前后半对应)配对，二者共享同一组角度 (c_j, s_j)。
    #   cos/sin(长 half)各复制一份拼成 rotary_dim，使前半第 j 位与后半第 j 位都拿到 c_j/s_j。
    c2 = torch.cat([c, c], dim=-1)                   # 前后半同角，拼到 head_dim
    s2 = torch.cat([s, s], dim=-1)
    # _rotate_half(rot)=[-后半, 前半]，逐元素展开后：
    #   前半第 j 位 = a_j·c_j - b_j·s_j；后半第 j 位 = b_j·c_j + a_j·s_j
    #   (a=前半, b=后半) —— 即对每对 (a_j,b_j) 做标准 2D 旋转，与 interleave 同式、仅配对/摆放不同。
    return x * c2 + _rotate_half(x) * s2
