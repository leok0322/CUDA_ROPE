"""QK-Norm + RoPE 的 PyTorch 参考实现。

FusedQKVNormRope: 合并 qkv [num_tokens, (Hq+Hk+Hv)*head_dim]，仅 Q/K 头做 RMSNorm
                  (分离 q/k 权重) + RoPE、V 透传，贴近真实 vLLM kernel。
  - RMSNorm 作用于每头整个 head_dim；RoPE 仅作用于前 rotary_dim 维(默认 neox，可选 interleave)。
  - 核心数学(rms_norm_pure / rope_pure / _rotate_half)统一收在 util.py(单一真相源)，
    本文件经 util.<fn> 调用；同一份函数也供 rmsnorm_rope_replace_pass.search_fn 复用，避免漂移。
  - RoPE 的 cos/sin cache 通过 util.build_cos_sin_cache 构造(布局与 CUDA kernel 一致)。
"""
import torch
import torch.nn as nn

try:                       # 兼容"包内导入"与"同目录脚本导入"两种方式
    from . import util
except ImportError:
    import util


# =============================================================================
# 升级版：合并 qkv + 仅 QK 处理 + 分离 q/k 权重（贴近真实 vLLM kernel 的形态）
# =============================================================================
class FusedQKVNormRope(nn.Module):
    """对【合并的 qkv】做 QK-Norm + RoPE，V 段透传（不归一化、不旋转）。

    约定输入 qkv: [num_tokens, (Hq + Hk + Hv) * head_dim]
      每个 token 内按 [Q 头… | K 头… | V 头…] 连续排布，每头 head_dim 个元素
      （与 src/inference/fused_qknorm_rope_kernel.cu 的 qkv 布局一致）。
    - 仅 Q、K 头做 RMSNorm(QK-Norm) + RoPE；V 头原样透传。
    - Q、K 各用一份独立的 γ：q_weight / k_weight（均 [head_dim]）。
    forward(qkv, positions)：positions 缺省 arange(num_tokens)。
    """

    def __init__(self, num_heads_q, num_heads_k, num_heads_v, head_dim,
                 rotary_dim=None, max_position=8192, eps=1e-6, base=10000.0,
                 interleave=False, dtype=torch.float32, device="cpu"):
        # 必须在任何 self.xxx = Parameter/子模块 赋值【之前】调用：
        #   nn.Module.__init__ 会建好 _parameters/_buffers/_modules 等内部簿记字典，
        #   之后赋的 self.weight(Parameter)、register_buffer 的 cos_sin_cache 才会被登记，
        #   从而 parameters()/.to()/state_dict() 等机制正常工作；漏掉它，下面赋
        #   nn.Parameter 时会直接抛 AttributeError(cannot assign parameter before
        #   Module.__init__() call)。用 super() 而非 nn.Module.__init__(self)：随 MRO
        #   解析父类，不写死基类名、并支持多继承链式初始化。
        super().__init__()
        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.num_heads_v = num_heads_v
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim if rotary_dim is not None else head_dim
        assert self.rotary_dim % 2 == 0, "rotary_dim 必须为偶数"
        assert self.rotary_dim <= head_dim, "rotary_dim 不能大于 head_dim"
        self.half = self.rotary_dim // 2
        self.eps = eps
        self.interleave = interleave

        # Q、K 各自独立的 RMSNorm 缩放 γ（长度 = head_dim），全 1 初始化
        
        # RMSNorm 的可学习缩放 γ（长度 = dim）。主流写法(对齐 nn.LayerNorm)：
        #   先用 torch.empty 声明参数(只占位、不决定数值)，再由 reset_parameters()
        #   统一用 nn.init.ones_ 初始化为全 1（全 1 → 初始等价于纯归一化）。
        #   把初始化收口到 reset_parameters()，便于复用/重置参数。
        self.q_weight = nn.Parameter(torch.empty(head_dim, dtype=dtype, device=device))
        self.k_weight = nn.Parameter(torch.empty(head_dim, dtype=dtype, device=device))
        self.reset_parameters()


        # 通过 util.build_cos_sin_cache 构造 cos/sin cache，注册为 buffer。
        # 注册成 buffer(而非普通属性)：它非可训练参数(无梯度)，但仍需随
        #   .to(device)/.cuda()/.half() 自动搬迁——普通 self.xxx 是搬不动的。
        # persistent=False：仍随模块迁移设备/精度，但【不写入 state_dict、不随
        #   checkpoint 存盘/加载】。因为本 cache 是由 max_position/rotary_dim/base
        #   纯计算【可重建】的派生常量：存进 checkpoint 既浪费体积，又会在日后改这些
        #   超参时因形状/键不匹配而冲突；不存它，加载旧 checkpoint(无此键)在 strict
        #   模式下也不会报 missing key。(反例：BatchNorm 的 running_mean 无法凭超参
        #   重算，必须 persistent=True。)
        cache = util.build_cos_sin_cache(
            max_position, self.rotary_dim, base=base, dtype=dtype, device=device)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def reset_parameters(self):
        """初始化/重置可学习参数：RMSNorm 缩放 γ 置全 1（PyTorch 标准做法）。"""
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)

    def _qk_norm_rope(self, x, weight, cos, sin):
        """对一组头(Q 或 K)：先整头 RMSNorm，再【仅前 rotary_dim 维做 RoPE】，其余透传。
        x: [num_tokens, num_heads, head_dim]。rotary_dim<head_dim 时才有 pass_through。"""
        x = util.rms_norm_pure(x, weight, self.eps)          # RMSNorm 作用于【整个 head_dim】
        rot = x[..., :self.rotary_dim]                  # 参与旋转的前 rotary_dim 维
        pass_through = x[..., self.rotary_dim:]         # 超出 rotary_dim 的高维不旋转
        rot = util.rope_pure(rot, cos, sin, self.interleave)
        return torch.cat([rot, pass_through], dim=-1)

    def forward(self, qkv, positions=None):
        Hq, Hk, Hv = self.num_heads_q, self.num_heads_k, self.num_heads_v
        H = Hq + Hk + Hv
        num_tokens = qkv.shape[0]
        assert qkv.shape[-1] == H * self.head_dim, \
            f"qkv 最后一维应为 {H*self.head_dim}=(Hq+Hk+Hv)*head_dim"

        # 1) [num_tokens, H*head_dim] → [num_tokens, H, head_dim]，按头切 Q/K/V
        v = qkv.view(num_tokens, H, self.head_dim)
        q = v[:, :Hq]                       # [num_tokens, Hq, head_dim]
        k = v[:, Hq:Hq + Hk]                # [num_tokens, Hk, head_dim]
        v_pass = v[:, Hq + Hk:]             # [num_tokens, Hv, head_dim] —— 透传

        # 2) 取 cos/sin（gather 在此处、即 op 外完成 → 方案 A）
        if positions is None:
            positions = torch.arange(num_tokens, device=qkv.device)
        cos = self.cos_sin_cache[positions, :self.half].to(qkv.dtype)   # [num_tokens, half]
        sin = self.cos_sin_cache[positions, self.half:].to(qkv.dtype)

        # 3) 仅 Q、K：各自 RMSNorm(分离权重) → 前 rotary_dim 维 RoPE(其余透传)；V 不动
        q = self._qk_norm_rope(q, self.q_weight, cos, sin)
        k = self._qk_norm_rope(k, self.k_weight, cos, sin)

        # 4) 拼回并摊平回 [num_tokens, H*head_dim]
        out = torch.cat([q, k, v_pass], dim=1)
        return out.reshape(num_tokens, H * self.head_dim)

# 注：方案 A 的待匹配子图(search_fn)已由 RMSNormRoPEreplacePass.search_fn 实现(复用本文件的
#     rms_norm_pure / rope_pure，且含 rot/pass_through 切分以支持 rotary_dim<head_dim)；
#     原先这里的独立 fused_qkv_search_neox 已删除，避免与模型 forward 逻辑重复/漂移。
