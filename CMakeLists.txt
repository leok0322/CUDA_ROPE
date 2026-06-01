cmake_minimum_required(VERSION 3.20)

# 必须在 project() 之前设置，锁定编译器路径，避免多版本共存时误用错误版本
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CXX_COMPILER  "/usr/bin/g++")

# 声明项目名称并启用 CXX 和 CUDA 两种语言
#   .cpp → g++ 编译为纯主机代码（interface.cpp 中的 pybind11 模块定义）
#   .cu  → nvcc 编译为主机 + 设备混合代码（run_kernel.cu / validation.cu / runner.cu 中的 CUDA kernel）
project(CUDA_ROPE LANGUAGES CXX CUDA)

# 导出编译命令到 compile_commands.json，供 clangd / clang-tidy 做代码分析和补全
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 现代 CMake 方式（CMake 3.17+）查找 CUDA Toolkit
#   提供 IMPORTED target：CUDA::cudart / CUDA::cublas 等
find_package(CUDAToolkit REQUIRED)

# 查找 Python3 解释器和开发头文件
#   Development 组件提供 Python3_INCLUDE_DIRS（Python.h）和 Python3_LIBRARIES
#   torch/extension.h → torch/csrc/python_headers.h → <Python.h>，需要 Development 组件
#   通过 cmake -DPython3_ROOT_DIR=<venv路径> 或 -DPython3_EXECUTABLE=<python3路径> 锁定 venv
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")

# ── GPU 架构检测（必须在 find_package(Torch) 之前）────────────────────────────
# 必须在 find_package(Torch) 之前完成的原因：
#   find_package(Torch) 执行时，PyTorch 内部的 cuda.cmake 会读取
#   TORCH_CUDA_ARCH_LIST 来决定为哪些 GPU 架构配置 LibTorch 的 CUDA 部分。
#   依赖链的顺序是：
#     execute_process(nvidia-smi) → _GPU_CC = "8.6"
#           ↓
#     set(TORCH_CUDA_ARCH_LIST "${_GPU_CC}")   ← 必须在 find_package(Torch) 之前设好
#           ↓
#     find_package(Torch)  ← PyTorch cmake 在此阶段读取 TORCH_CUDA_ARCH_LIST
#   如果检测步骤放在 find_package(Torch) 之后，TORCH_CUDA_ARCH_LIST 为空，
#   PyTorch 要么报错，要么用错误的默认架构，导致编译出的 kernel 在当前 GPU 上
#   无法运行或性能异常。
#
# OUTPUT_STRIP_TRAILING_WHITESPACE：去掉末尾换行，否则 _GPU_CC 为 "8.6\n"
# ERROR_QUIET：nvidia-smi 不存在时静默失败，_GPU_CC 保持为空
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _GPU_CC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# 将 "8.6" → "86"：nvcc 的 -gencode 参数需要无小数点的整数形式
string(REPLACE "." "" _GPU_CC_NUM "${_GPU_CC}")

if(NOT _GPU_CC_NUM)
    message(FATAL_ERROR "无法检测 GPU 架构，请确认 nvidia-smi 可用")
endif()

# ── find_package(Torch) 前的准备 ─────────────────────────────────────────────
# TORCH_CUDA_ARCH_LIST 须在 find_package(Torch) 之前设置，供 PyTorch cmake 读取
# unset CMAKE_CUDA_ARCHITECTURES：project(LANGUAGES CUDA) 初始化时会自动赋默认值，
#   不 unset 则 find_package(Torch) 触发 "not compatible with CMAKE_CUDA_ARCHITECTURES" 警告
set(TORCH_CUDA_ARCH_LIST "${_GPU_CC}")
unset(CMAKE_CUDA_ARCHITECTURES CACHE)

# 查找 LibTorch（PyTorch 的 C++ 后端）
#   需通过 -DCMAKE_PREFIX_PATH=<torch cmake 目录> 指定搜索路径，在 CLion CMake options 中添加：
#   -DCMAKE_PREFIX_PATH=/home/liam/python_linux/python_venv/.venv/lib/python3.13/site-packages/torch/share/cmake
find_package(Torch REQUIRED)

# find_package(Torch) 执行后的三个已知 Warning（均可忽略）：
#   Warning 1: Failed to compute shorthash for libnvrtc.so     —— 本项目不使用 NVRTC，无影响
#   Warning 2: pytorch is not compatible with CMAKE_CUDA_ARCHITECTURES —— 已通过 unset 修复
#   Warning 3: static library kineto_LIBRARY-NOTFOUND not found —— 此安装包未含 Kineto，无影响

# CMAKE_CUDA_ARCHITECTURES 须在 find_package(Torch) 之后、check_language(CUDA) 之前设置
# CMake 将其展开为 nvcc 的 -gencode arch=compute_86,code=[sm_86,compute_86]
set(CMAKE_CUDA_ARCHITECTURES ${_GPU_CC_NUM})
message(STATUS "GPU Compute Capability: ${_GPU_CC} -> CMAKE_CUDA_ARCHITECTURES=${_GPU_CC_NUM}  TORCH_CUDA_ARCH_LIST=${_GPU_CC}")

# 查找 libtorch_python.so：PYBIND11_MODULE 展开时需要，find_package(Torch) 未包含此子库
find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED
)

# 探测 CUDA 编译环境是否完整可用（实际编译一段最小 CUDA 代码验证）
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found")
endif()

# PyTorch 2.x 要求至少 C++17；C++20 作为超集向后兼容
set(CMAKE_CXX_STANDARD 20)
# 不支持 C++20 时直接报错，而非静默降级
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 可配置的 kernel 编译参数 ──────────────────────────────────────────────────
# 三层优先级（高 → 低）：
#   1. cmake -DROPE_VARIANT=2        → 写入 cache，最高优先级
#   2. set(ROPE_VARIANT 1 CACHE ...) → cache 无值时的 cmake 层默认值
#   3. #ifndef ROPE_VARIANT          → build.ninja 无 -D 时的编译器层兜底
#      #define ROPE_VARIANT 1
# 注意：修改此文件中的默认值不会覆盖 CMakeCache.txt 中的已有值，
#       需删除 CMakeCache.txt 或显式传 -D 才能让修改生效
set(ROPE_VARIANT 1 CACHE STRING "RoPE kernel variant (1-N), matches kernel implementation")
set(BLOCK_SIZE   256 CACHE STRING "CUDA block size (threads per block), power of 2 in [32, 1024]")

message(STATUS "Kernel config: ROPE_VARIANT=${ROPE_VARIANT}  BLOCK_SIZE=${BLOCK_SIZE}")

# ── 全局编译选项 ──────────────────────────────────────────────────────────────
# $<$<AND:...>:flag> 是生成器表达式，配置阶段求值，按条件决定是否写入 build.ninja

# Debug 模式：-G 为 GPU 设备代码生成调试符号，供 cuda-gdb 单步调试
#   副作用：禁用几乎所有 GPU 优化，不可用于性能测量
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")

# Release 模式：-lineinfo 保留源码行号，让 ncu 能将性能数据映射回源码行
#   不禁止优化，Release 下 ncu 分析才能反映真实性能
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>")

# Release 模式：--use_fast_math 用 GPU 硬件内置近似函数替代标准数学函数
#   展开为：-ftz=true（denormal 刷零）/ -prec-div=false / -prec-sqrt=false
add_compile_options("$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--use_fast_math>")

# 两种模式均开启：--ptxas-options=-v 输出每个 kernel 的寄存器 / 共享内存用量
#   影响 occupancy 分析，两种模式均需查看
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>")

# 将 TORCH_CXX_FLAGS（如 "-D_GLIBCXX_USE_CXX11_ABI=1"）从字符串拆分为 CMake list
#   必须拆分：直接传字符串会被当作单个含空格的参数，编译器无法解析
#   拆分后每项独立传给编译器，ABI 标志与 libtorch.so 保持一致，避免链接时 undefined reference
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS_LIST})

# ── ROPE_cuda 共享库（Python 扩展模块）───────────────────────────────────────
# 构建为 SHARED 库，Python import 时由解释器 dlopen 加载
# 编译流程：
#   g++  src/interface.cpp  → interface.cpp.o   （pybind11 模块定义，纯主机代码）
#   nvcc src/run_kernel.cu  → run_kernel.cu.o   （RoPE kernel 实现，含 SASS + PTX）
#   ld -shared → ROPE_cuda.cpython-313-x86_64-linux-gnu.so
add_library(ROPE_cuda SHARED
    src/interface.cpp
    src/run_kernel.cu
)

# 查询 venv 解释器的 Python C 扩展标准后缀，如 .cpython-313-x86_64-linux-gnu.so
#   Python import 时按固定规则搜索文件名，不符合规范则报 ModuleNotFoundError
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE _PY_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# 设置共享库文件名：去掉 CMake 默认的 "lib" 前缀，并替换后缀为 Python 标准扩展名
#   默认：libROPE_cuda.so → 修改后：ROPE_cuda.cpython-313-x86_64-linux-gnu.so
if(_PY_EXT_SUFFIX)
    set_target_properties(ROPE_cuda PROPERTIES PREFIX "" SUFFIX "${_PY_EXT_SUFFIX}")
    message(STATUS "Python extension suffix: ${_PY_EXT_SUFFIX}")
else()
    set_target_properties(ROPE_cuda PROPERTIES PREFIX "")
    message(WARNING "无法检测 Python 扩展后缀，生成裸 .so；若需直接 import，请手动重命名")
endif()

# 将 CMake 变量写成编译器 -D 标志，注入为 C 预处理器宏
#   cmake -DROPE_VARIANT=2 → CMake 变量 "2" 写入 cache
#   target_compile_definitions → build.ninja 中追加 -DROPE_VARIANT=2
#   nvcc 编译时预处理：#ifndef ROPE_VARIANT → 已定义，跳过默认值，使用传入的 2
#
#   TORCH_EXTENSION_NAME：PYBIND11_MODULE 宏展开时使用此名称作为 Python 模块名
#     必须与库文件名（ROPE_cuda）一致，否则 import 时 PyInit_ROPE_cuda 符号找不到
target_compile_definitions(ROPE_cuda PRIVATE
    TORCH_EXTENSION_NAME=ROPE_cuda
    ROPE_VARIANT=${ROPE_VARIANT}
    BLOCK_SIZE=${BLOCK_SIZE}
)

target_include_directories(ROPE_cuda PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}   # cuda_runtime.h 等 CUDA 运行时头文件
    ${TORCH_INCLUDE_DIRS}         # torch/extension.h / pybind11 头文件
    ${Python3_INCLUDE_DIRS}       # Python.h（torch/extension.h 间接需要）
)

target_link_libraries(ROPE_cuda PRIVATE
    CUDA::cudart              # CUDA 运行时（cudaMalloc / kernel 调度等）
    ${TORCH_LIBRARIES}        # libtorch.so / libtorch_cpu.so / libc10.so 等
    ${TORCH_PYTHON_LIBRARY}   # libtorch_python.so，PYBIND11_MODULE 展开时需要
)

set_target_properties(ROPE_cuda PROPERTIES
    CXX_STANDARD  20   # interface.cpp 用 C++20 编译
    CUDA_STANDARD 20   # run_kernel.cu 设备代码也用 C++20，无全局变量可继承，必须显式设置
    # 输出到项目根目录，Python 脚本可直接 import，无需修改 sys.path
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# ── validation 可执行文件（C++ 侧 kernel 基准测试程序）────────────────────────
# 与 ROPE_cuda（SHARED 库）的区别：
#   ROPE_cuda  : 无 main()，由 python3 dlopen 加载
#   validation : 有 main()，./validation <kernel_num> 直接运行
#
# 编译流程：
#   nvcc src/validation.cu                          → validation.cu.o   （主程序：参数解析、计时、正确性验证）
#   nvcc src/runner.cu                              → runner.cu.o        （kernel 启动函数定义）
#   nvcc src/inference/fused_qknorm_rope_kernel.cu  → fused_qknorm_rope_kernel.cu.o （fused QK Norm + RoPE）
#   ld → validation
#
# validation.cu 通过 #include "runner.cuh" 看到函数声明，
# 链接时从 runner.cu.o 找到实际定义，不加 runner.cu 则报 undefined reference
add_executable(validation
    src/validation.cu
    src/runner.cu
    src/inference/fused_qknorm_rope_kernel.cu   # fused QK Norm + RoPE kernel（stable ABI）
)

target_compile_definitions(validation PRIVATE USE_CUDA)

target_include_directories(validation PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}              # cuda_runtime.h / cudaEvent_t / cublas_v2.h 等
    /usr/include/x86_64-linux-gnu            # cudnn.h：apt 安装 cuDNN 时头文件位于此路径
                                             # /usr/local/cuda/include 不含 cudnn.h，需单独指定
    # ${PROJECT_SOURCE_DIR}/src/inference    # 无需添加：fused_qknorm_rope_kernel.cu 本身位于 src/inference/，
                                             # quoted include（"torch_utils.h" 等）优先搜索当前文件所在目录，
                                             # 本地头文件自动可见
    # ${TORCH_INCLUDE_DIRS}                  # 无需显式添加：下方 target_link_libraries 链接的
                                             # torch imported target 已把这两个目录写入其
                                             # INTERFACE_INCLUDE_DIRECTORIES（TorchConfig.cmake:156），
                                             # 经 target_link_libraries(validation PRIVATE torch) 传播给
                                             # 本目标的编译命令，torch 头文件路径自动可见：
                                             #   torch/headeronly/util/Exception.h（STD_TORCH_CHECK）
                                             #   torch/csrc/stable/tensor.h（torch::stable::Tensor）
                                             #   torch/csrc/inductor/aoti_torch/c/shim.h 等 stable ABI 头文件
                                             # 注：USE_CUDA 宏不在 torch 接口属性里，仍须上方显式
                                             #     target_compile_definitions 定义，不能靠链 torch 获得。
)

target_link_libraries(validation PRIVATE
    CUDA::cudart   # cudaMalloc / cudaFree / cudaMemcpy / cudaEventCreate 等运行时符号
    cudnn          # cuDNN 运行时库（libcudnn.so）：kernel_base 使用 cuDNN RoPE 参考实现
    torch          # PyTorch libtorch 核心 imported target，提供 aoti_torch_get_current_cuda_stream
                   # 等 stable ABI 符号（torch_utils.h 中 get_current_cuda_stream 所需）。
                   # 该符号实际定义在 libtorch_cuda.so，而非 libtorch.so：
                   #   torch 的 INTERFACE_LINK_LIBRARIES = "torch_cpu_library;torch_cuda_library"
                   #   torch_cuda_library 用 -Wl,--no-as-needed 强制链入 libtorch_cuda.so，
                   #   故只链 torch 即可传递拉到该符号；libtorch_cuda.so 的 NEEDED
                   #   又会自动带出 libtorch_cpu.so / libc10.so 等运行时依赖。
                   # 无需用 ${TORCH_LIBRARIES}（含 c10/caffe2/kineto/fbgemm 等一长串），
                   #   本目标用不到，链 torch 一个目标即足够且更精简。
)

set_target_properties(validation PROPERTIES
    CXX_STANDARD  20
    CUDA_STANDARD 20
    # 输出到项目根目录，run.sh 中 EXEC="./validation" 可直接找到
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
