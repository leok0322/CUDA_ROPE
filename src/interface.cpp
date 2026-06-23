#include <torch/extension.h>

// ───────────────────────────────────────────────────────────────────────────
// 真正的 kernel 入口 fused_QKNorm_and_ROPE_interleave 定义在 run_kernel.cu（由 nvcc 编译）。
// 此处仅【前向声明】，链接期解析到 run_kernel.cu 中的定义。
//
// 这是【写法③：void in-place】—— 就地改写 qkv、返回 void。配套约束：
//   - 前向声明的签名必须和真实 kernel 一致，且【把全部参数列全】(qkv/权重/cos/sin/各头数/eps)。
//   - qkv 在 schema 里要标 (a!)(见下方 TORCH_LIBRARY)；C++ 这里用引用 at::Tensor& 表达"就地改写"。
//   - 方案 A：cos/sin 已在算子外按 position 取好([num_tokens, rotary_dim/2])，不收 cache/position_ids。
//   - num_tokens 不作参数：内部由 qkv.size(0) 推得；qkv.size(1) 应 == (Hq+Hk+Hv)*head_dim。
//
// 关于 at::Tensor vs torch::Tensor：二者是【同一个类型】——torch 头文件里有 `using at::Tensor;`
//   故 torch::Tensor 只是 at::Tensor 的别名(static_assert(is_same_v<...>) 成立)，C++ 签名里两者
//   可互换(本文件用 at:: 风格；换成 torch::Tensor 一字不差也能编译)。
//   ⚠ 但这只限【C++ 签名】；下方 schema 字符串(m.def)里【只能写 Tensor】，
//      写 at::Tensor / torch::Tensor 都会解析失败。
// ───────────────────────────────────────────────────────────────────────────
void fused_QKNorm_and_ROPE_interleave(
    at::Tensor& qkv,                  // [num_tokens, (Hq+Hk+Hv)*head_dim]  ★就地改写
    const at::Tensor& q_weight,       // [head_dim]  Q 的 RMSNorm γ
    const at::Tensor& k_weight,       // [head_dim]  K 的 RMSNorm γ
    const at::Tensor& cos,            // [num_tokens, rotary_dim/2]  已 gather
    const at::Tensor& sin,            // [num_tokens, rotary_dim/2]
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, int64_t rotary_dim, double eps);  // rotary_dim: 旋转维度(<=head_dim)，half=rotary_dim/2

// CUDA 后端封装函数：算子分发到 CUDA 时实际执行它，内部转调真正的 kernel 入口。
// 形参类型必须与 schema 对应(见下方“schema 类型 ↔ C++ 类型”说明)：
//   schema int   ↔ C++ int64_t（不是 uint/int）
//   schema float ↔ C++ double （torch schema 的 float 就是 64 位 double，不是 C++ float）
//   schema Tensor(a!) ↔ at::Tensor&（就地改写）
// 写法③(void in-place)：本函数返回 void、把结果就地写回 qkv；Python 侧 replace_fn 以
//   op(qkv,...); return qkv 把被改写的 qkv 交出去(算子本身无返回)。
void run_fused_QKNorm_and_ROPE_cuda_interleave(
    at::Tensor qkv, at::Tensor q_weight, at::Tensor k_weight,
    at::Tensor cos, at::Tensor sin,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, int64_t rotary_dim, double eps) {
    // ★ 必须把【全部参数转发】给真正的 kernel，否则 kernel 收不到数据、什么也算不了。
    fused_QKNorm_and_ROPE_interleave(qkv, q_weight, k_weight, cos, sin,
                                     num_heads_q, num_heads_k, num_heads_v,
                                     head_dim, rotary_dim, eps);
}

// ───────────────────────────────────────────────────────────────────────────
// 用 TORCH_LIBRARY 把本 .so 注册为 torch 自定义算子库。
//   - TORCH_LIBRARY      ：声明算子 schema（算子名 + 函数签名），即"声明"。
//   - TORCH_LIBRARY_IMPL ：为 CUDA 分发键(dispatch key)注册具体实现，即"分配到 cuda"。
//
// 命名分两层（注意区分，二者不是同一个东西）：
//   命名空间(namespace) = TORCH_LIBRARY 的第一个参数，这里是 ROPE_cuda
//   算子名(op name)     = m.def(...) schema 字符串里的函数名，这里是 fused_qkv_norm_rope_interleave
//   调用形式：torch.ops.<命名空间>.<算子名>()
//            torch.ops.ROPE_cuda.fused_qkv_norm_rope_interleave(...)
//                      └ namespace ┘ └──────── op name ────────┘
//
// 与 .so 文件名的关系：命名空间和算子名【都不需要】与 .so 文件名相同。
//   原理：TORCH_LIBRARY / TORCH_LIBRARY_IMPL 展开为【静态注册器】，在 .so 被
//        torch.ops.load_library() 以 dlopen 加载的那一刻自动运行，把算子按上面的
//        宏参数名登记到全局 dispatcher——登记名来自宏参数，与文件名无关。
//   对比：PYBIND11_MODULE 必须让"模块名 = .so 文件名"（import 时找 PyInit_<name>）；
//        torch.ops 机制靠 load_library 给路径即可，文件名随意，故二者解耦。
//
// Python 侧使用方式（torch.ops 机制，而非 import 扩展模块）：
//   torch.ops.load_library("任意路径/任意名.so")   # 文件名随意
//   torch.ops.ROPE_cuda.fused_qkv_norm_rope_interleave(...)
//
// 【.so 路径要求 —— torch.ops 方式】
//   - 位置：任意目录均可，无需放进 sys.path / PYTHONPATH，更不需放进 torch 安装目录；
//           load_library() 直接给【绝对或相对路径】即可。
//   - 文件名：任意（与命名空间/算子名解耦）。
//   - 运行期：先 `import torch`，使其依赖的 libtorch*/libc10* 等先入进程，
//             你的 .so 被 dlopen 时符号才解析得了（否则需 RPATH 或 LD_LIBRARY_PATH）。
// ───────────────────────────────────────────────────────────────────────────
TORCH_LIBRARY(ROPE_cuda, m) {   // ROPE_cuda = 命名空间(namespace)
    // m.def 的作用：声明算子的【接口/契约(schema)】——它是参数类型/顺序/返回的【唯一真相源】。
    //   Python 不能直达本 C++ 函数：调用先经【通用 Dispatcher】(把实参装箱成 IValue、按算子名+
    //   分发键 CPU/CUDA/Meta 路由)，Dispatcher 不认识具体 C++ 签名，只认这份 schema 来装箱/校验/路由。
    //   分工：m.def=声明接口(一次)；m.impl(下方)=注册实现(可多个，模板自动拆箱 IValue→C++ 类型调函数)。
    //   故一个算子可挂多实现(CUDA kernel / Meta-fake / Autograd)共享同一 schema —— 这是 m.def 不可省的原因。
    // schema 是【自己的 DSL】，不是 C++ 类型名，几个易错点：
    //   - 张量写 Tensor（不是 torch::Tensor）；
    //   - 整数写 int（= C++ int64_t；没有 uint）；浮点写 float（= C++ double）；
    //   - 【就地改写的张量必须标 (a!)】：Tensor(a!) qkv —— 告诉编译器 qkv 会被改写，
    //     functionalization 才会用 auto_functionalized 正确处理；漏了 (a!) → 编译器当纯算子 →
    //     void 无输出被 DCE(kernel 不跑) / 缓冲区错乱(静默错误)。
    //   - "-> ()" 表示无返回(写法③ void in-place)。
    //   - 算子名【必须与 Python 侧 op_name 一字一致】：runner.py 用
    //     ROPE_cuda::fused_qkv_norm_rope_interleave，故这里也叫 fused_qkv_norm_rope_interleave，
    //     否则 _resolve_op 的 getattr 找不到 → AttributeError。
    m.def("fused_qkv_norm_rope_interleave(Tensor(a!) qkv, Tensor q_weight, Tensor k_weight, "
          "Tensor cos, Tensor sin, int num_heads_q, int num_heads_k, int num_heads_v, "
          "int head_dim, int rotary_dim, float eps) -> ()");
    // 注：neox 风格需另注册一个 fused_qkv_norm_rope_neox(签名相同)，由 Runner 据 interleave 选用。
}

TORCH_LIBRARY_IMPL(ROPE_cuda, CUDA, m) {   // 同一命名空间 ROPE_cuda 的 CUDA 实现
    // 把算子名 fused_qkv_norm_rope_interleave(须与上面 m.def 的一致)的 CUDA 实现，
    // 绑定到 C++ 封装函数 run_fused_QKNorm_and_ROPE_cuda_interleave。
    m.impl("fused_qkv_norm_rope_interleave", &run_fused_QKNorm_and_ROPE_cuda_interleave);
}

// ───────────────────────────────────────────────────────────────────────────
// 另外再提供 PYBIND11_MODULE 绑定：支持 Python 直接 import 调用
// （与上面的 torch.ops 机制【并存】，两种调用方式皆可，按需选用）。
//   - 模块名用 TORCH_EXTENSION_NAME（CMake 中定义为 ROPE_cuda）。与 torch.ops 不同，
//     PYBIND11 要求【模块名 = .so 文件名】，import 时解释器据此找 PyInit_ROPE_cuda 符号。
//   - m.def 暴露一个 Python 函数，名为 fused_qkv_norm_rope_interleave，指向 C++ 封装
//     run_fused_QKNorm_and_ROPE_cuda_interleave。
//
// Python 侧两种用法对照：
//   方式一(torch.ops)：torch.ops.load_library("任意名.so")
//                       torch.ops.ROPE_cuda.fused_qkv_norm_rope_interleave(...)
//   方式二(import)    ：import ROPE_cuda          # 需 .so 文件名 = ROPE_cuda.*.so
//                       ROPE_cuda.fused_qkv_norm_rope_interleave(...)
//
// 【.so 路径要求 —— import 方式】
//   - 位置：.so 所在目录必须在 sys.path 上。Python 启动时自动把
//           "脚本所在目录(python a.py)"或"cwd(REPL / -c / -m)"加入 sys.path[0]；
//           本项目 CMake 把 .so 输出到项目根(LIBRARY_OUTPUT_DIRECTORY=PROJECT_SOURCE_DIR)，
//           故脚本放在根、或从根目录起交互/-c/-m 即可直接 import。
//           想从任意位置 import：设 PYTHONPATH、sys.path.append(...)、或装进 site-packages。
//           （同样【不需要】放进 torch 安装目录。）
//   - 文件名：必须是 ROPE_cuda.<EXT_SUFFIX>.so（EXT_SUFFIX 即 SOABI，
//             如 .cpython-313-x86_64-linux-gnu.so），import 才认得出模块名 ROPE_cuda；
//             由 CMake 的 Python3_add_library(... WITH_SOABI) 自动保证。裸 libROPE_cuda.so 不行。
//   - 运行期：同上，先 `import torch` 让 libtorch* 依赖就位。
// ───────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // pybind 用的是 C++ 类型(不受 schema DSL 限制)；函数名与 torch.ops 那套保持一致即可。
    m.def("fused_qkv_norm_rope_interleave", &run_fused_QKNorm_and_ROPE_cuda_interleave,
          "Fused QKV-Norm and RoPE, interleave (CUDA, in-place on qkv)");
}

// ═══════════════════════════════════════════════════════════════════════════
// 【对照参考：改用 stable ABI(torch::stable::Tensor) 的写法】
//
// 为什么用 #if 0 包起来(默认不编译)：
//   1) 同命名空间不能重复注册同名算子——上方已用 TORCH_LIBRARY(ROPE_cuda,...) 注册了
//      fused_qkv_norm_rope_interleave，stable 版若同名同命名空间会 duplicate registration
//      报错。故此处改用【不同命名空间 ROPE_cuda_stable】示意。
//   2) stable ABI 的接口(boxed 调用约定 / 宏名 / 辅助)随 PyTorch 版本演进，启用前请按
//      你的版本核对头文件与签名。标 ⚠ 处尤需确认。
//
// 经典(上方)  →  stable(下方) 的对应关系：
//   #include <torch/extension.h>        →  stable/headeronly 头(见下)
//   at::Tensor& / at::Tensor            →  torch::stable::Tensor& / torch::stable::Tensor
//   TORCH_LIBRARY / _IMPL               →  STABLE_TORCH_LIBRARY / _IMPL
//   TORCH_CHECK                         →  STD_TORCH_CHECK
//   x.is_cuda()/x.size()/x.data_ptr()   →  stable 版的等价接口(见 fused_qk_norm_rope_kernel.cu)
//   schema 字符串                        →  不变(仍写 Tensor / Tensor(a!) / -> ())
// ═══════════════════════════════════════════════════════════════════════════
#if 0
// ⚠ 头文件按版本核对；典型为：
#include <torch/csrc/stable/library.h>   // STABLE_TORCH_LIBRARY / _IMPL
#include <torch/csrc/stable/tensor.h>    // torch::stable::Tensor
#include <torch/headeronly/macros/Macros.h>  // STD_TORCH_CHECK 等

// CUDA 封装：与经典版逻辑相同，仅把 at::Tensor 换成 torch::stable::Tensor。
// 真正的 kernel 入口 fused_QKNorm_and_ROPE_interleave 仍是上方那个(收 at::Tensor)；
// stable 张量需先桥接成 ATen 张量再转调，或另写一个收 stable 张量的 kernel 入口。
void run_fused_QKNorm_and_ROPE_cuda_interleave_stable(
    torch::stable::Tensor qkv, torch::stable::Tensor q_weight,
    torch::stable::Tensor k_weight, torch::stable::Tensor cos,
    torch::stable::Tensor sin,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, int64_t rotary_dim, double eps) {
    // ⚠ 这里需要把 torch::stable::Tensor 转成 kernel 能用的形式(data_ptr/形状)，
    //   或经官方桥接转 at::Tensor 后调用 fused_QKNorm_and_ROPE_interleave(...)。
    //   具体接口随版本不同，参见 src/inference/fused_qknorm_rope_kernel.cu 里
    //   torch::stable::Tensor 的用法(get_device_index / DeviceGuard / data_ptr 等)。
}

// schema 字符串与经典版完全一致(Tensor(a!) 标就地改写、float=double、-> () 无返回)。
STABLE_TORCH_LIBRARY(ROPE_cuda_stable, m) {   // ← 不同命名空间，避免与 ROPE_cuda 冲突
    m.def("fused_qkv_norm_rope_interleave(Tensor(a!) qkv, Tensor q_weight, Tensor k_weight, "
          "Tensor cos, Tensor sin, int num_heads_q, int num_heads_k, int num_heads_v, "
          "int head_dim, int rotary_dim, float eps) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(ROPE_cuda_stable, CUDA, m) {
    // ⚠ stable ABI 走 boxed 调用约定：部分版本可直接 m.impl("name", &fn)，
    //   部分版本需用 TORCH_BOX(&fn) / 手写 boxed 包装(StableIValue* stack,...)。按版本核对。
    m.impl("fused_qkv_norm_rope_interleave",
           &run_fused_QKNorm_and_ROPE_cuda_interleave_stable);
}
// Python 侧调用：torch.ops.ROPE_cuda_stable.fused_qkv_norm_rope_interleave(...)
#endif
