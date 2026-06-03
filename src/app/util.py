import os

import torch
import yaml

# YAML 里 dtype 是字符串，这里映射到真正的 torch.dtype
_DTYPE_MAP = {
    "float32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


def load_config(path=None):
    """读取 YAML 配置文件，返回可直接 Runner(**cfg) 展开的配置字典。

    path 缺省为本文件同目录下的 config/config.yaml。dtype 字段是字符串，经 _DTYPE_MAP
    映射到 torch.dtype；device 为 null 时保留 None，交给 Runner.__init__ 自动选。
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
    # dtype 字符串 → torch.dtype
    if "dtype" in cfg:
        cfg["dtype"] = _DTYPE_MAP[str(cfg["dtype"]).lower()]
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


def make_inputs(seed,num_tokens,head_num,head_dim,device,dtype,token_size,batch):
    """随机 3D 输入 x: [num_tokens, head_num, head_dim]，及每 token 的 position。"""
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, head_num, head_dim,
                            device=device, dtype=dtype)
    # 每条序列各自 0..token_size-1 的位置，再按 batch 拼接 → [num_tokens]
    positions = torch.arange(token_size, device=device).repeat(batch)
    return x, positions
