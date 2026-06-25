"""rmsnorm_rope_replace_pass.py —— 定义"换什么"：合并 qkv 的 QK-Norm+RoPE 分步子图(search)
   与融合算子调用(replace)，以及描摹该子图形状的样例张量(example_inputs)结构。

针对【FusedQKVNormRope】(合并 qkv + 仅 QK 处理 + 分离 q/k 权重；方案 A：cos/sin 在 op 外
已 gather)。只负责"匹配-替换的内容定义"，不负责"如何安装进 Inductor"(那是 Installer 的职责)。

────────────────────────────────────────────────────────────────────────────
设计要点
────────────────────────────────────────────────────────────────────────────
- 张量通配符(matcher 绑定) = (qkv, q_weight, k_weight, cos, sin)，共 5 个；cos/sin 是
  【已 gather】的 [num_tokens, half](gather 留在 model.forward 即 op 外，pattern 不含
  cache 索引/arange → 方案 A)。
- num_heads_q/k/v、head_dim、eps 是【常量】：search 端被 trace 时固定下来，replace 端作为
  算子调用实参烤入；is_neox/interleave 不作算子入参，而是 Runner 按其值选不同 op_name。
- search_fn 复用 util 的纯函数(rms_norm_pure / rope_pure)，与 FusedQKVNormRope.forward
  逐步同构，避免重复实现导致漂移。
- example_inputs 的【结构】由 search_fn 决定，故留在本类；运行期规格(num_tokens/device/dtype)
  由 Runner 作参数传入，不存为构造字段。
────────────────────────────────────────────────────────────────────────────
"""
import torch

# 复用 util 的纯函数，保证 search 子图与模型 forward 逐步同构（单一真相源、不漂移）
try:
    from .util import rms_norm_pure, rope_pure
except ImportError:
    from util import rms_norm_pure, rope_pure


class RMSNormRoPEreplacePass:
    """合并 qkv 的 QK-Norm+RoPE 子图 → 融合算子 的 search / replace 定义。"""

    def __init__(self, eps, num_heads_q, num_heads_k, num_heads_v, head_dim,
                 op_name, interleave, rotary_dim):
        self.eps = eps                       # search(rsqrt 常量) + replace(op 入参)
        self.num_heads_q = num_heads_q       # Q 头数 Hq
        self.num_heads_k = num_heads_k       # K 头数 Hk
        self.num_heads_v = num_heads_v       # V 头数 Hv（透传，不处理）
        self.head_dim = head_dim
        # rotary_dim<head_dim 时每头只旋转前 rotary_dim 维、其余透传。
        # 【必传】：由 Runner 统一解析(None→head_dim)并校验后传入，此处不再兜底，保证 model 与
        #   search_fn 用同一 rotary_dim(子图才对得上)；去掉死代码默认值，契约更显式。
        self.rotary_dim = rotary_dim
        self.half = self.rotary_dim // 2
        # 融合算子全名(命名空间::算子名)，供 Installer 注册 fake/meta 实现用
        self.op_name = op_name
        # RoPE 风格：决定 search_fn 匹配 neox(False) 还是 interleave(True) 子图，
        # 须与 model.py 里 FusedQKVNormRope 的 interleave 一致，否则计算图对不上、匹配不到。
        self.interleave = interleave

    # --------------------------------- 单组头(Q 或 K)的 RMSNorm + RoPE（与模型一致）
    def _qk_norm_rope(self, x, weight, cos, sin):
        """整头 RMSNorm，再【仅前 rotary_dim 维做 RoPE】，其余透传。
        x: [num_tokens, num_heads, head_dim]。复用 model 纯函数，逐步同构 forward。"""
        x = rms_norm_pure(x, weight, self.eps)          # RMSNorm 作用于整个 head_dim
        rot = x[..., :self.rotary_dim]                  # 参与旋转的前 rotary_dim 维
        pass_through = x[..., self.rotary_dim:]         # 超出 rotary_dim 的高维不旋转
        rot = rope_pure(rot, cos, sin, self.interleave)
        return torch.cat([rot, pass_through], dim=-1)

    # --------------------------------- 待匹配子图(search)：合并 qkv，仅 QK 处理
    def search_fn(self, qkv, q_weight, k_weight, cos, sin):
        """与 FusedQKVNormRope.forward 逐步同构的待匹配子图。
        张量通配符 = (qkv, q_weight, k_weight, cos, sin)；cos/sin 已 gather [num_tokens, half]。
        Hq/Hk/Hv、head_dim、eps、interleave 在 trace 时作为常量固定。
        注：interleave 是 Python 常量布尔(经 rope_pure)，register_replacement 描摹时按它选定
        一支(neox / interleave)，只有被选中的那套算子序列进入待匹配图(不会留下未走的分支)。"""
        Hq, Hk, Hv = self.num_heads_q, self.num_heads_k, self.num_heads_v
        H = Hq + Hk + Hv
        num_tokens = qkv.shape[0]
        v = qkv.view(num_tokens, H, self.head_dim)      # [num_tokens, H, head_dim]
        q = v[:, :Hq]                                   # Q 头
        k = v[:, Hq:Hq + Hk]                            # K 头
        v_pass = v[:, Hq + Hk:]                         # V 头（透传）
        q = self._qk_norm_rope(q, q_weight, cos, sin)
        k = self._qk_norm_rope(k, k_weight, cos, sin)
        out = torch.cat([q, k, v_pass], dim=1)
        return out.reshape(num_tokens, H * self.head_dim)

    # --------------------------------- 替换目标(replace)：一次融合算子调用
    def _resolve_op(self):
        """按 op_name("命名空间::算子名")取出 torch.ops 里的算子可调用对象。

        op_name 是运行期字符串(随 interleave 在 Runner 里动态选)，不能用写死的点号访问，
        故用 getattr 做"字符串→属性"的动态查找。等价于 torch.ops.<ns>.<name>：
          ns, name = "ROPE_cuda::fused_qkv_norm_rope_neox".split("::")
          内层 getattr(torch.ops, ns)  → torch.ops.ROPE_cuda (命名空间对象)
          外层 getattr(...,      name) → ....fused_qkv_norm_rope_neox (算子可调用对象)
        注：torch.ops 的 ns/name 不是普通 Python 属性，不在任何 __dict__ 里；
            它们重写了 __getattr__，把字符串当【Dispatcher 注册表的键】动态解析——
            该键即 C++ TORCH_LIBRARY(ns)+m.def("name(...)->...") 注册的全名，
            所以 op_name 必须与 C++ 注册的 ns::name 逐字一致，否则查不到、抛 AttributeError。"""
        ns, name = self.op_name.split("::")
        return getattr(getattr(torch.ops, ns), name)

    def replace_fn(self, qkv, q_weight, k_weight, cos, sin):
        """替换成一次合并 qkv 融合算子调用。

        注意签名仍是 (qkv, q_weight, k_weight, cos, sin) —— 与 search_fn 一致，这 5 个是
        pattern 的【张量通配符】(matcher 绑定的就是它们)。而 num_heads_q/k/v / head_dim / eps
        是【配置常量/可从形状推】，不进 pattern，作为常量从 self 读出、烤进【算子调用实参】：

            op(qkv, q_weight, k_weight, cos, sin, num_heads_q, num_heads_k, num_heads_v,
               head_dim, rotary_dim, eps)

        is_neox/interleave 不作为算子入参，而是【在 Runner 里按其值选不同的算子】
        (Runner 据 interleave 传入不同的 op_name)，此处只按 op_name 解析出对应算子。

        ★ 依赖：interface.cpp 里对应算子的 schema 需为
           <op_name>(Tensor(a!) qkv, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin,
                     int num_heads_q, int num_heads_k, int num_heads_v,
                     int head_dim, int rotary_dim, float eps) -> ()
           且 neox / interleave 各注册一个算子；否则此调用会 schema 不匹配。
           (rotary_dim：旋转维度<=head_dim，kernel 据此算 half=rotary_dim/2、定旋转/透传区。)"""
        op = self._resolve_op()
        # 前 5 个是【张量通配符】(matcher 绑定)，后 5 个是【常量】(从 self 烤入)：
        #   qkv      : [num_tokens, (Hq+Hk+Hv)*head_dim]   合并 qkv，每 token 内 [Q…|K…|V…]
        #   q_weight : [head_dim]                          Q 的 RMSNorm γ
        #   k_weight : [head_dim]                          K 的 RMSNorm γ
        #   cos      : [num_tokens, rotary_dim//2]         已 gather(方案 A)
        #   sin      : [num_tokens, rotary_dim//2]
        #   num_heads_q/k/v, head_dim : int 常量(切 Q/K/V、定形状)
        #   rotary_dim : int 常量(旋转维度<=head_dim；kernel 据此算 half、定旋转/透传区)
        #   eps      : float 常量(RMSNorm 数值稳定项)
        op(qkv, q_weight, k_weight, cos, sin,          # [nt,H*hd] [hd] [hd] [nt,r/2] [nt,r/2]
                self.num_heads_q, self.num_heads_k, self.num_heads_v,  # int Hq, Hk, Hv
                self.head_dim, self.rotary_dim, self.eps)   # int head_dim, int rotary_dim, float eps
        return qkv

    # --------------------------------- 本算子的 fake/meta（随 op 语义而定，注入给 Installer 注册）
    def make_fake_fn(self):
        """返回本算子的 fake/meta 实现（供 Installer 用 register_fake(op_name, fake) 注册）。
        fake 的"返回什么"由算子 schema 决定，故由【拥有算子的本类】给出，而非 Installer 写死：
          - 本算子是【写法③ void in-place（-> ()）】→ 无输出张量 → 返回 None；
          - 若改函数式（-> Tensor），应改成 lambda qkv, *a: torch.empty_like(qkv)。
        *args 通配：fake 只声明返回、不读参数(参数流入见 installer.py 注释)。"""

        # 融合算子需要 fake/meta 实现，torch.compile 才能在 FakeTensor 上推断形状。
        # 写法③(void in-place)：算子 -> ()，故 fake 返回 None(不另开输出)。
        # 形参用 *args 通配：fake 只声明“返回 None”、不读任何参数，无需逐个命名(避免名字对不上语义)。
        # 参数流入路径(非直接调用，编译期间接到达)：
        #   make_example_inputs(5 张量) → example_inputs → register_replacement trace replace_fn
        #   → replace_fn 调 op(qkv,q_weight,k_weight,cos,sin, Hq,Hk,Hv,head_dim,eps)
        #   →(FakeTensorMode 路由到本 fake) _fake(*args)：前 5=example_inputs(转 FakeTensor)、
        #     后 5=replace_fn 从 self 烤入的常量。(trace search_fn 不调本 fake)
        def _fake(*args):
            return None
        return _fake

    # --------------------------------- 样例张量(结构归本类、运行期规格外部传入)
    def make_example_inputs(self, num_tokens, device, dtype):
        """供 register_replacement 把 search_fn/replace_fn trace 成 FX pattern 的样例张量。
        【只返回 5 个张量】(与 search_fn/replace_fn 形参一一对应=pattern 通配符)；num_heads_q/k/v、
        head_dim、eps 是常量、由 replace_fn 从 self 烤入算子，【不放进 example_inputs】(否则
        search_fn(*example_inputs) 参数个数对不上而报错)。只需形状/精度对、数值随机；与运行期
        模型输入(util.make_inputs_qkv)无关。详见 docs/.../torch.PatternMatcherPass_register_replacement.txt 五点五。
        结构：qkv[num_tokens,(Hq+Hk+Hv)*head_dim]、q/k_weight[head_dim]、cos/sin[num_tokens,half]。"""
        H = self.num_heads_q + self.num_heads_k + self.num_heads_v
        return [
            torch.randn(num_tokens, H * self.head_dim, device=device, dtype=dtype),  # qkv
            torch.randn(self.head_dim, device=device, dtype=dtype),  # q_weight γ
            torch.randn(self.head_dim, device=device, dtype=dtype),  # k_weight γ
            torch.randn(num_tokens, self.half, device=device, dtype=dtype),  # cos
            torch.randn(num_tokens, self.half, device=device, dtype=dtype),  # sin
        ]
