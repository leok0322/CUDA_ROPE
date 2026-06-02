#include <torch/extension.h>

// ───────────────────────────────────────────────────────────────────────────
// 真正的 kernel 入口 fused_QKNorm_and_ROPE 定义在 run_kernel.cu（由 nvcc 编译）。
// 此处仅【前向声明】，链接期解析到 run_kernel.cu 中的定义。
// ───────────────────────────────────────────────────────────────────────────
void fused_QKNorm_and_ROPE();

// CUDA 后端封装函数：算子分发到 CUDA 时实际执行它，内部转调真正的 kernel 入口。
void run_fused_QKNorm_and_ROPE_cuda() {
    fused_QKNorm_and_ROPE();
}

// ───────────────────────────────────────────────────────────────────────────
// 用 TORCH_LIBRARY 把本 .so 注册为 torch 自定义算子库。
//   - TORCH_LIBRARY      ：声明算子 schema（算子名 + 函数签名），即"声明"。
//   - TORCH_LIBRARY_IMPL ：为 CUDA 分发键(dispatch key)注册具体实现，即"分配到 cuda"。
//
// 命名分两层（注意区分，二者不是同一个东西）：
//   命名空间(namespace) = TORCH_LIBRARY 的第一个参数，这里是 ROPE_cuda
//   算子名(op name)     = m.def(...) schema 字符串里的函数名，这里是 run_fused_QKNorm_and_ROPE
//   调用形式：torch.ops.<命名空间>.<算子名>()
//            torch.ops.ROPE_cuda.run_fused_QKNorm_and_ROPE()
//                      └ namespace ┘ └──── op name ─────┘
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
//   torch.ops.ROPE_cuda.run_fused_QKNorm_and_ROPE()
//
// 【.so 路径要求 —— torch.ops 方式】
//   - 位置：任意目录均可，无需放进 sys.path / PYTHONPATH，更不需放进 torch 安装目录；
//           load_library() 直接给【绝对或相对路径】即可。
//   - 文件名：任意（与命名空间/算子名解耦）。
//   - 运行期：先 `import torch`，使其依赖的 libtorch*/libc10* 等先入进程，
//             你的 .so 被 dlopen 时符号才解析得了（否则需 RPATH 或 LD_LIBRARY_PATH）。
// ───────────────────────────────────────────────────────────────────────────
TORCH_LIBRARY(ROPE_cuda, m) {   // ROPE_cuda = 命名空间(namespace)
    // schema 声明：run_fused_QKNorm_and_ROPE 是【算子名】；
    // "() -> ()" 表示空入参、空返回。
    m.def("run_fused_QKNorm_and_ROPE() -> ()");
}

TORCH_LIBRARY_IMPL(ROPE_cuda, CUDA, m) {   // 同一命名空间 ROPE_cuda 的 CUDA 实现
    // 把算子 run_fused_QKNorm_and_ROPE（算子名，须与上面 m.def 的一致）的 CUDA
    // 实现绑定到 C++ 封装函数 run_fused_QKNorm_and_ROPE_cuda。
    m.impl("run_fused_QKNorm_and_ROPE", &run_fused_QKNorm_and_ROPE_cuda);
}

// ───────────────────────────────────────────────────────────────────────────
// 另外再提供 PYBIND11_MODULE 绑定：支持 Python 直接 import 调用
// （与上面的 torch.ops 机制【并存】，两种调用方式皆可，按需选用）。
//   - 模块名用 TORCH_EXTENSION_NAME（CMake 中定义为 ROPE_cuda）。与 torch.ops 不同，
//     PYBIND11 要求【模块名 = .so 文件名】，import 时解释器据此找 PyInit_ROPE_cuda 符号。
//   - m.def 暴露一个 Python 函数，名为 run_fused_QKNorm_and_ROPE，指向 C++ 封装
//     run_fused_QKNorm_and_ROPE_cuda。
//
// Python 侧两种用法对照：
//   方式一(torch.ops)：torch.ops.load_library("任意名.so")
//                       torch.ops.ROPE_cuda.run_fused_QKNorm_and_ROPE()
//   方式二(import)    ：import ROPE_cuda          # 需 .so 文件名 = ROPE_cuda.*.so
//                       ROPE_cuda.run_fused_QKNorm_and_ROPE()
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
    m.def("run_fused_QKNorm_and_ROPE", &run_fused_QKNorm_and_ROPE_cuda,
          "Fused QK-Norm and RoPE (CUDA)");
}
