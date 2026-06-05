"""compiled.py —— torch.compile 子图替换的【通用安装器】。

本类不关心具体匹配什么子图，只负责把外部传入的
  search_fn  (要被匹配的分步子图)
  replace_fn (替换成的融合算子调用)
  example_inputs (供 register_replacement 描摹子图形状的样例张量)
注册成一条 register_replacement 规则，塞进 PatternMatcherPass，
再挂到 Inductor 的 post_grad 钩子(对应 docs/fork/fused_silu_mul_compile_hook.py 第 3~4 步)。

search_fn / replace_fn / example_inputs 的具体定义留在 Runner，
Runner 构造 Compiled 时把它们注入进来 —— 职责：Runner 定义"换什么"，Compiled 负责"怎么装"。
"""
import torch


class Installer:
    """通用的 "匹配-替换" pass 安装器：注入 search/replace/example_inputs 即可用。"""

    def __init__(self, search_fn, replace_fn, example_inputs,
                 op_name="", fake_fn=None):
        self.search_fn = search_fn          # 被匹配的分步子图
        self.replace_fn = replace_fn        # 替换成的融合算子调用
        self.example_inputs = example_inputs  # 样例张量，供 register_replacement 描摹形状
        self.op_name = op_name              # 融合算子全名，用于注册 fake/meta 实现
        # fake/meta 实现【由调用方(拥有算子的 pass)提供并注入】，本类不写死——因为"fake 返回什么"
        # 取决于算子 schema(void→None / 函数式→empty_like)，是算子语义的一部分。
        self.fake_fn = fake_fn

    # ----------------------- 注册替换 → PatternMatcherPass → post_grad 钩子
    def install_fusion_pass(self):
        import torch._inductor.config as inductor_config
        from torch._inductor.pattern_matcher import (
            PatternMatcherPass, register_replacement, fwd_only,
        )

        # 融合算子需要 fake/meta 实现，torch.compile 才能在 FakeTensor 上推断形状。
        # fake 实现【由调用方注入(self.fake_fn)】，本类只负责"注册"，不写死其返回逻辑。
        # 何时被调(参数流入路径，非直接调用、编译期间接到达)：
        #   make_example_inputs(5 张量) → example_inputs → register_replacement trace replace_fn
        #   → replace_fn 调 op(qkv,q_weight,k_weight,cos,sin, Hq,Hk,Hv,head_dim,eps)
        #   →(FakeTensorMode 路由到 fake_fn) fake_fn(*args)：前 5=example_inputs(转 FakeTensor)、
        #     后 5=replace_fn 从 self 烤入的常量。(trace search_fn 不调它)
        if self.fake_fn is not None:
            torch.library.register_fake(self.op_name, self.fake_fn)

        pm_pass = PatternMatcherPass()
        register_replacement(self.search_fn, self.replace_fn,
                             self.example_inputs, fwd_only, pm_pass)

        def custom_post_grad_pass(graph):
            count = pm_pass.apply(graph)
            if count:
                print(f"[hook] 融合发生：替换了 {count} 处 (RMSNorm+RoPE) → {self.op_name}")

        inductor_config.post_grad_custom_post_pass = custom_post_grad_pass
        return pm_pass
