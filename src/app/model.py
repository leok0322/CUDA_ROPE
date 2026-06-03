"""fused_QKNorm_ROPE 的 PyTorch 参考实现。

对输入 3D 张量 x：
  1) 先对【最后一维】做 RMSNorm；
  2) 再对【最后一维】做 RoPE（默认 neox 风格，可选 interleave）。
RoPE 的 cos/sin cache 通过 util.build_cos_sin_cache 构造（布局与 CUDA kernel 一致）。
"""
import torch
import torch.nn as nn

try:                       # 兼容"包内导入"与"同目录脚本导入"两种方式
    from . import util
except ImportError:
    import util


def _rotate_half(x):
    """neox 风格旋转辅助：把最后一维分成前后两半 [x1, x2] -> [-x2, x1]。"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class QKNorm_ROPE(nn.Module):
    """对输入 3D 张量 x 依次做：最后一维 RMSNorm → 最后一维 RoPE。

    约定输入形状 x: [num_tokens, num_heads, head_dim]
      - 最后一维 head_dim：被 RMSNorm 归一化、被 RoPE 旋转的【特征维】；
      - 第 0 维 num_tokens：【位置维】，每个 token 一个 position（用于查 cos/sin）；
      - 中间维 num_heads：RoPE 中按位置广播（同一 token 的各 head 共享 cos/sin）。
    forward(x, positions=None)：positions 缺省为 arange(num_tokens)。
    """

    def __init__(self, dim, rotary_dim=None, max_position=8192,
                 eps=1e-6, base=10000.0, interleave=False,
                 dtype=torch.float32, device="cpu"):
        # 必须在任何 self.xxx = Parameter/子模块 赋值【之前】调用：
        #   nn.Module.__init__ 会建好 _parameters/_buffers/_modules 等内部簿记字典，
        #   之后赋的 self.weight(Parameter)、register_buffer 的 cos_sin_cache 才会被登记，
        #   从而 parameters()/.to()/state_dict() 等机制正常工作；漏掉它，下面赋
        #   nn.Parameter 时会直接抛 AttributeError(cannot assign parameter before
        #   Module.__init__() call)。用 super() 而非 nn.Module.__init__(self)：随 MRO
        #   解析父类，不写死基类名、并支持多继承链式初始化。
        super().__init__()
        self.dim = dim
        self.rotary_dim = rotary_dim if rotary_dim is not None else dim
        assert self.rotary_dim % 2 == 0, "rotary_dim 必须为偶数"
        assert self.rotary_dim <= dim, "rotary_dim 不能大于最后一维 dim"
        self.half = self.rotary_dim // 2
        self.eps = eps
        self.interleave = interleave

        # RMSNorm 的可学习缩放 γ（长度 = dim）。主流写法(对齐 nn.LayerNorm)：
        #   先用 torch.empty 声明参数(只占位、不决定数值)，再由 reset_parameters()
        #   统一用 nn.init.ones_ 初始化为全 1（全 1 → 初始等价于纯归一化）。
        #   把初始化收口到 reset_parameters()，便于复用/重置参数。
        self.weight = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
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
        nn.init.ones_(self.weight)

    # ---- 步骤 1：最后一维 RMSNorm ----
    def _rms_norm(self, x):
        in_dtype = x.dtype
        x = x.float()                                       # 在 float32 上算更稳
        # 均方值 ms = mean(x^2)（RMSNorm 的核心统计量，即二阶原点矩 E[X^2]）：
        #   x.pow(2)         逐元素平方，形状不变 [.., head_dim]
        #   .mean(dim=-1)    沿最后一维(特征维 head_dim)求平均，对每个 (token,head) 行
        #                    取该行 head_dim 个平方值的均值；【不减均值】(区别于 LayerNorm)
        #   keepdim=True     保留末维为 1 → [.., 1]，便于下一步与 x 广播相除
        ms = x.pow(2).mean(dim=-1, keepdim=True)            # 二阶原点矩：mean(x^2) over 最后一维
        # x/[..,head_dim] 乘 rsqrt(ms)/[..,1]：靠广播自动把末维 1 拉伸到 head_dim，
        #   即每行 head_dim 个特征同乘一个标量(该行 RMS 的倒数)。依赖上面 keepdim=True
        #   保留末维为 1；否则形状 [..,head_dim]*[..] 从右对齐失败需手动 unsqueeze(-1)。
        #   广播是逻辑扩展、不复制数据。
        x = x * torch.rsqrt(ms + self.eps)                  # x / sqrt(mean(x^2)+eps)
        return x.to(in_dtype) * self.weight                 # 逐维乘 γ

    # ---- 步骤 2：最后一维 RoPE ----
    def _apply_rope(self, x, cos, sin):
        # x: [num_tokens, num_heads, dim]; cos/sin: [num_tokens, half]
        rot = x[..., :self.rotary_dim]                      # 参与旋转的前 rotary_dim 维
        pass_through = x[..., self.rotary_dim:]             # 超出 rotary_dim 的高维不旋转
        # 索引里 None = 在该位置插一个 size=1 的新维(等价 unsqueeze(1))，: 表示该维原样保留。
        #   cos: [num_tokens, half] -> [num_tokens, 1, half]，中间补出的就是 head 维。
        #   cos/sin 本无 head 维：RoPE 旋转角只与 position 和频率 i 有关、与是哪个 head 无关，
        #   故同一 token 的所有 head 共享同一组角度。补出长度 1 后，与 x[num_tokens,num_heads,..]
        #   相乘时这个 1 沿 head 维广播成 num_heads，即每个 head 都乘同一份 cos/sin。
        cos = cos[:, None, :]                               # [num_tokens, 1, half]，在 head 维广播
        sin = sin[:, None, :]
        if self.interleave:
            # interleave(=!neox)：相邻 (2i, 2i+1) 配对
            # 切片 [..., start::step]：... 保留前面所有维、只切最后一维(rotary_dim)；
            #   0::2 = 从下标0起步长2 → 偶数位 0,2,4,…；1::2 = 从1起步长2 → 奇数位 1,3,5,…
            #   即把相邻一对 (2i,2i+1) 分别抽进 x1[i]=rot[2i]、x2[i]=rot[2i+1]，各 half 长。
            x1 = rot[..., 0::2]                             # 偶数位 -> [.., half]
            x2 = rot[..., 1::2]                             # 奇数位 -> [.., half]
            # 每对 (x1,x2) 用同一角度 θ 旋转：r1=x1cosθ-x2sinθ, r2=x1sinθ+x2cosθ
            # cos/sin 是 [.,1,half]，与 x1[.,num_heads,half] 相乘时中间 1 广播成 num_heads，
            # 故 r1、r2 形状均为 [num_tokens, num_heads, half]。
            r1 = x1 * cos - x2 * sin
            r2 = x1 * sin + x2 * cos
            # torch.stack：在【新建维度】上叠同形状张量(区别于 cat 沿已有维拼接，不增维)。
            #   dim=-1 把长度2的新维插到最后：[.,num_heads,half] -> [.,num_heads,half,2]，
            #   末维 [r1[i], r2[i]] 即每个频率位的一对。
            # flatten(-2)：合并倒数两维 half*2=rotary_dim，按最内先变展开成
            #   r1[0],r2[0],r1[1],r2[1],… 的【交错】排布(若用 cat 会得到前半全r1/后半全r2，
            #   那是 neox 排布，不是 interleave)。
            rot = torch.stack([r1, r2], dim=-1).flatten(-2)
        else:
            # neox：第 j 维 ↔ 第 j+half 维(前后半对应)配对，二者共享同一组角度 (c_j, s_j)。
            #   cos/sin(长 half)各复制一份拼成 rotary_dim，使前半第 j 位与后半第 j 位都拿到 c_j/s_j。
            cos2 = torch.cat([cos, cos], dim=-1)           # [.., rotary_dim] 前后半同角
            sin2 = torch.cat([sin, sin], dim=-1)
            # _rotate_half(rot)=[-后半, 前半]，逐元素展开后：
            #   前半第 j 位 = a_j·c_j - b_j·s_j；后半第 j 位 = b_j·c_j + a_j·s_j
            #   (a=前半, b=后半) —— 即对每对 (a_j,b_j) 做标准 2D 旋转，与 interleave 同式、仅配对/摆放不同。
            rot = rot * cos2 + _rotate_half(rot) * sin2
        return torch.cat([rot, pass_through], dim=-1)

    def forward(self, x, positions=None):
        assert x.dim() == 3, "输入必须是 3D 张量 [num_tokens, num_heads, head_dim]"
        assert x.shape[-1] == self.dim, f"最后一维应为 {self.dim}，实际 {x.shape[-1]}"
        num_tokens = x.shape[0]

        # 1) 最后一维 RMSNorm
        x = self._rms_norm(x)

        # 2) 最后一维 RoPE
        if positions is None:
            positions = torch.arange(num_tokens, device=x.device)
        cos = self.cos_sin_cache[positions, :self.half]    # [num_tokens, half]
        sin = self.cos_sin_cache[positions, self.half:]    # [num_tokens, half]
        x = self._apply_rope(x, cos.to(x.dtype), sin.to(x.dtype))
        return x
