"""rmsnorm_rope_replace_pass.py —— 定义"换什么"：RMSNorm+RoPE 分步子图(search) 与融合算子调用(replace)，
   以及描摹该子图形状的样例张量(example_inputs)结构。

只负责"匹配-替换的内容定义"，不负责"如何安装进 Inductor"(那是 Installer 的职责)。

────────────────────────────────────────────────────────────────────────────
职责划分结论（构造入参为何只留 5 个；example_inputs 为何留在本类但参数外注入）
────────────────────────────────────────────────────────────────────────────
对“匹配子图(被 trace 成 pattern 的 search 图)”而言，各参数影响如下：
  - 影响匹配子图：interleave(切换算子序列结构)、eps(烤成 rsqrt(ms+eps) 的字面常量)、
                  dtype(经 example_inputs 的 dtype 间接决定 .float()/.to() cast 节点有无)。
  - 不影响匹配子图：num_tokens / head_num / head_dim / half / device（只定 example_inputs
                  形状/设备，pattern 匹配对维度形状无关）；op_name(只作用在 replace/fake 侧)。

构造函数只持有【真正服务 search_fn / replace_fn 的长期状态】：
  - search_fn 用：eps、interleave
  - replace_fn 用：op_name、head_num、head_dim、eps（作为常量烤进算子调用实参）
故构造函数收缩为 (eps, head_num, head_dim, op_name, interleave)。

而 example_inputs 的【结构】(几个张量、哪个是 x/weight/cos/sin、各几维)由 search_fn 决定，
与“要匹配的子图”强耦合，故 make_example_inputs 仍留在本类(放 Runner 会割裂内聚)；
但它依赖的【运行期张量规格】(num_tokens / device / dtype)不是 pattern 语义、属 Runner 所有，
所以【作为方法参数由 Runner 传入】，而非存成构造函数的长期字段(half 还冗余 = head_dim//2)。
边界：
  Runner       管运行期形状/设备/精度，并把它们传给 make_example_inputs
  本类         管 pattern 语义(search) + 替换目标(replace) + example_inputs 的结构
  Installer    管安装(register_replacement + PatternMatcherPass + post_grad 钩子)
────────────────────────────────────────────────────────────────────────────
"""
import torch


class RMSNormRoPEreplacePass:
    """RMSNorm+RoPE 分步子图 → 融合算子 的 search / replace 定义（不含 example_inputs）。"""

    def __init__(self, eps, head_num, head_dim, op_name, interleave):
        self.eps = eps               # search_fn(rsqrt 常量) + replace_fn(op 入参)
        self.head_num = head_num     # replace_fn 的 op 入参(num_heads)
        self.head_dim = head_dim     # replace_fn 的 op 入参(head_dim)
        # 融合算子全名(命名空间::算子名)，供 Installer 注册 fake/meta 实现用
        self.op_name = op_name
        # RoPE 风格标志：决定 search_fn 匹配 neox(False) 还是 interleave(True) 子图，
        # 须与 model.py 里 QKNorm_ROPE 的 interleave 一致，否则计算图对不上、匹配不到。
        self.interleave = interleave

    # ----------------------------------- 分步(search) 与 融合算子(replace)
    @staticmethod
    def _rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    def _rms_norm(self, x, weight):
        """两种风格共用的最后一维 RMSNorm。"""
        in_dtype = x.dtype
        xf = x.float()
        ms = xf.pow(2).mean(-1, keepdim=True)          # 二阶原点矩 mean(x^2)
        xf = xf * torch.rsqrt(ms + self.eps)           # x / sqrt(mean(x^2)+eps)
        return xf.to(in_dtype) * weight                # 乘 RMSNorm 权重 γ

    def _search_neox(self, x, weight, cos, sin):
        """neox 风格(interleave=False)子图：RMSNorm 后前后半配对旋转。"""
        xn = self._rms_norm(x, weight)
        c = cos[:, None, :]                            # [num_tokens,1,half] 在 head 维广播
        s = sin[:, None, :]
        c2 = torch.cat([c, c], dim=-1)                 # 拼到 head_dim 长
        s2 = torch.cat([s, s], dim=-1)
        return xn * c2 + self._rotate_half(xn) * s2

    def _search_interleave(self, x, weight, cos, sin):
        """interleave 风格(interleave=True)子图：RMSNorm 后相邻 (2i,2i+1) 配对旋转。
        与 model.py 的 if-interleave 分支(0::2/1::2 + stack/flatten)逐步对应。"""
        xn = self._rms_norm(x, weight)
        c = cos[:, None, :]                            # [num_tokens,1,half]
        s = sin[:, None, :]
        x1 = xn[..., 0::2]                             # 偶数位
        x2 = xn[..., 1::2]                             # 奇数位
        r1 = x1 * c - x2 * s
        r2 = x1 * s + x2 * c
        return torch.stack([r1, r2], dim=-1).flatten(-2)

    def search_fn(self, x, weight, cos, sin):
        """要被匹配的子图：最后一维 RMSNorm，然后按 self.interleave 选 RoPE 风格。
        与 model.py 的 _rms_norm + _apply_rope（rotary_dim==head_dim）逐步对应。
        cos/sin: [num_tokens, head_dim/2]。
        注：self.interleave 是 Python 常量布尔，register_replacement 描摹时按它选定一支，
        只有被选中的那套算子序列进入待匹配图(不会留下未走的分支)。"""
        if self.interleave:
            return self._search_interleave(x, weight, cos, sin)
        return self._search_neox(x, weight, cos, sin)

    def _resolve_op(self):
        """按 op_name("命名空间::算子名")取出 torch.ops 里的算子可调用对象。

        op_name 是运行期字符串(随 interleave 在 Runner 里动态选),不能用写死的点号访问，
        故用 getattr 做"字符串→属性"的动态查找。等价于 torch.ops.<ns>.<name>：
          ns, name = "ROPE_cuda::run_fused_QKNorm_and_ROPE_neox".split("::")
          内层 getattr(torch.ops, ns)  → torch.ops.ROPE_cuda (命名空间对象)
          外层 getattr(...,      name) → ....run_..._neox     (算子可调用对象)
        注：torch.ops 的 ns/name 不是普通 Python 属性，不在任何 __dict__ 里；
            它们重写了 __getattr__，把字符串当【Dispatcher 注册表的键】动态解析——
            该键即 C++ TORCH_LIBRARY(ns)+m.def("name(...)->...") 注册的全名，
            所以 op_name 必须与 C++ 注册的 ns::name 逐字一致，否则查不到、抛 AttributeError。"""
        ns, name = self.op_name.split("::")
        return getattr(getattr(torch.ops, ns), name)

    def replace_fn(self, x, weight, cos, sin):
        """替换成一次融合算子调用。

        注意签名仍是 (x, weight, cos, sin) —— 与 search_fn 一致，这 4 个是 pattern 的
        【张量通配符】(matcher 绑定的就是它们)。而 num_heads / head_dim / eps 是
        【配置常量/可从形状推】，不进 pattern，作为常量从 self 读出、烤进【算子调用实参】：

            op(x, weight, cos, sin, num_heads, head_dim, eps)

        is_neox/interleave 不作为算子入参，而是【在 Runner 里按其值选不同的算子】
        (Runner 据 interleave 传入不同的 op_name)，此处只按 op_name 解析出对应算子。

        ★ 依赖：interface.cpp 里对应算子的 schema 需为
           <op_name>(Tensor x, Tensor weight, Tensor cos, Tensor sin,
                     int num_heads, int head_dim, float eps) -> Tensor
           且 neox / interleave 各注册一个算子；否则此调用会 schema 不匹配。"""
        op = self._resolve_op()
        return op(x, weight, cos, sin, self.head_num, self.head_dim, self.eps)

    def make_example_inputs(self, num_tokens, device, dtype):
        """register_replacement 描摹子图形状用的样例张量(形状/精度须与真实输入一致)。

        【结构】由 search_fn 决定，故留在本类：x[num_tokens,head_num,head_dim]、
        weight[head_dim]、cos/sin[num_tokens,half]。
        【运行期规格】num_tokens/device/dtype 由 Runner 作参数传入(不存为本类字段)；
        half 由 head_dim 推(= head_dim//2)，不冗余存储。"""
        half = self.head_dim // 2
        return [
            torch.randn(num_tokens, self.head_num, self.head_dim,
                        device=device, dtype=dtype),                 # x
            torch.randn(self.head_dim, device=device, dtype=dtype),  # weight γ
            torch.randn(num_tokens, half, device=device, dtype=dtype),  # cos
            torch.randn(num_tokens, half, device=device, dtype=dtype),  # sin
        ]
