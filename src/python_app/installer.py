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
                 op_name=""):
        self.search_fn = search_fn          # 被匹配的分步子图
        self.replace_fn = replace_fn        # 替换成的融合算子调用
        self.example_inputs = example_inputs  # 样例张量，供 register_replacement 描摹形状
        self.op_name = op_name              # 融合算子全名，用于注册 fake/meta 实现

    # ----------------------- 注册替换 → PatternMatcherPass → post_grad 钩子
    def install_fusion_pass(self):
        import torch._inductor.config as inductor_config
        from torch._inductor.pattern_matcher import (
            PatternMatcherPass, register_replacement, fwd_only,
        )

        # 融合算子需要 fake/meta 实现，torch.compile 才能在 FakeTensor 上推断形状。
        # *args 吸收算子的常量入参(num_heads, head_dim, eps 等)，输出与输入 x 同形同类型。
        @torch.library.register_fake(self.op_name)
        def _fake(x, weight, cos, sin, *args):
            return torch.empty_like(x)

        pm_pass = PatternMatcherPass()
        register_replacement(self.search_fn, self.replace_fn,
                             self.example_inputs, fwd_only, pm_pass)

        def custom_post_grad_pass(graph):
            count = pm_pass.apply(graph)
            if count:
                print(f"[hook] 融合发生：替换了 {count} 处 (RMSNorm+RoPE) → "
                      f"torch.ops.ROPE_cuda.run_fused_QKNorm_and_ROPE")

        inductor_config.post_grad_custom_post_pass = custom_post_grad_pass
        return pm_pass
