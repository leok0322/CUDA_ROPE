# 从源码构建 flash-attention。
#
# 设计目标：
#   1. 参考 vLLM 的 external_projects/vllm_flash_attn.cmake，
#      用 FetchContent 把 flash-attention 的 CMake 项目纳入当前项目。
#   2. flash-attention 构建出来的 Python 扩展 / 共享库 .so，
#      在 build 阶段直接输出到本项目根目录 ${PROJECT_SOURCE_DIR}。
#      这样与 ROPE_cuda.cpython-*.so 的行为一致，Python 脚本可以在项目根目录
#      直接 import 或通过 torch.ops.load_library 加载。
#   3. 不修改本项目已有 ROPE_cuda / validation 的构建逻辑。

# 引入 CMake 官方 FetchContent 模块。
#
# FetchContent 用途：
#   - 配置阶段声明一个外部源码依赖；
#   - 可以从 GitHub 拉取，也可以指向本地源码目录；
#   - 拉取 / 准备完成后，用 FetchContent_MakeAvailable(...)
#     把外部项目的 CMakeLists.txt 加入当前构建图。
include(FetchContent)

# flash-attention 的 Git 仓库地址。
#
# CACHE STRING 的含义：
#   - 该变量写入 CMakeCache.txt；
#   - 用户可在 cmake 命令行用 -DFLASH_ATTENTION_GIT_REPOSITORY=...
#     覆盖默认值；
#   - STRING 表示这是普通字符串 cache 变量。
set(FLASH_ATTENTION_GIT_REPOSITORY
    "https://github.com/vllm-project/flash-attention.git"
    CACHE STRING "flash-attention Git repository")

# flash-attention 的 Git revision。
#
# 这里 pin 到 vLLM 示例中使用的 commit，避免每次配置都跟随 upstream 最新代码。
# 注意：
#   这里的变量名叫 GIT_TAG，但 CMake 的 FetchContent_Declare(GIT_TAG ...)
#   并不要求一定是真正的 Git tag；它可以是：
#       - branch 名，例如 main
#       - tag 名，例如 v2.6.2
#       - commit SHA，例如 b3964b1d8b95d8e8447435668ab169a2700bab65
#
# 本项目这里填写的是 commit SHA，不是 flash-attention 的 release version。
# 也就是说它不是 v2.6.2 / v2.7.0 这种版本号，而是精确固定到
# vllm-project/flash-attention 仓库中的某一次提交。
#
# 短 SHA：
#   b3964b1
#
# 这个 commit 来自 vLLM 使用的 flash-attention fork，
# 不是 upstream Dao-AILab/flash-attention 的普通 release tag。
# 例如 v2.6.2 这类 tag 会指向另一个 commit；这里没有使用 release tag，
# 而是直接使用 vLLM CMake 示例中固定的 commit，保证构建输入可复现。
#
# 固定 commit 的好处：
#   - 构建结果更可复现；
#   - 避免 flash-attention 主分支变化导致本项目突然构建失败。
set(FLASH_ATTENTION_GIT_TAG
    "b3964b1d8b95d8e8447435668ab169a2700bab65"
    CACHE STRING "flash-attention Git revision")

# 本地 flash-attention 源码路径。
#
# 默认空字符串，表示从 GitHub 下载。
# 如果用户设置：
#   -DFLASH_ATTENTION_SRC_DIR=/path/to/flash-attention
# 则不走 Git 下载，而是直接使用这个本地源码目录。
# 适合：
#   - 本地调试 flash-attention；
#   - 没有网络但已经提前 clone 源码；
#   - 修改 flash-attention 后让本项目直接使用本地版本。
set(FLASH_ATTENTION_SRC_DIR
    ""
    CACHE PATH "Use an existing flash-attention source directory instead of downloading")

# 环境变量优先于 CMake cache 变量。
#
# 这样做是为了方便临时覆盖：
#
#   FLASH_ATTENTION_SRC_DIR=/tmp/flash-attention cmake -S . -B build
#
# 不需要修改 CMakeCache.txt。
#
# 同时兼容 vLLM 的变量名 VLLM_FLASH_ATTN_SRC_DIR，方便复用 vLLM 开发习惯。
if(DEFINED ENV{FLASH_ATTENTION_SRC_DIR})
    set(FLASH_ATTENTION_SRC_DIR "$ENV{FLASH_ATTENTION_SRC_DIR}")
elseif(DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
    set(FLASH_ATTENTION_SRC_DIR "$ENV{VLLM_FLASH_ATTN_SRC_DIR}")
endif()

# 给 flash-attention / vLLM 风格 CMake 提供兼容变量。
#
# vLLM 的 flash-attention 构建脚本通常会依赖 VLLM_GPU_LANG 这类变量。
# 本项目只有 CUDA 后端，所以没有在顶层定义 VLLM_GPU_LANG。
# 如果上层没有定义，这里补一个默认值 "CUDA"。
if(NOT DEFINED VLLM_GPU_LANG)
    set(VLLM_GPU_LANG "CUDA")
endif()

# CUDA_ARCHS 是 vLLM 风格的架构列表，例如：
#   8.6
#   8.0;8.6;8.9
#
# 本项目顶层已有：
#   TORCH_CUDA_ARCH_LIST="${_GPU_CC}"   例如 "8.6"
#   CMAKE_CUDA_ARCHITECTURES=86
#
# 这里优先从 TORCH_CUDA_ARCH_LIST 推导 CUDA_ARCHS，因为它保留了小数点格式，
# 与 vLLM / flash-attention 常用的 arch 表达更接近。
# 如果没有 TORCH_CUDA_ARCH_LIST，再退回 CMAKE_CUDA_ARCHITECTURES。
if(NOT DEFINED CUDA_ARCHS)
    if(DEFINED TORCH_CUDA_ARCH_LIST)
        # TORCH_CUDA_ARCH_LIST 可能是空格分隔：
        #   "8.0 8.6"
        # CMake list 使用分号分隔：
        #   "8.0;8.6"
        # 因此这里把空格替换成分号，方便 foreach(IN LISTS ...) 遍历。
        string(REPLACE " " ";" CUDA_ARCHS "${TORCH_CUDA_ARCH_LIST}")
    elseif(DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# VLLM_GPU_ARCHES 是 vLLM flash-attention 期望的 CMake CUDA 架构格式，
# 例如：
#   86-real
#   80-real;86-real
#
# 本项目的 CUDA_ARCHS 可能是：
#   8.6        来自 TORCH_CUDA_ARCH_LIST
#   86         来自 CMAKE_CUDA_ARCHITECTURES
#   8.0+PTX    带 PTX 后缀的形式
#
# 这里把它统一转换为 "<无小数点数字>-real"。
# 示例：
#   8.6      -> 86-real
#   8.0+PTX  -> 80-real
#   86       -> 86-real
if(VLLM_GPU_LANG STREQUAL "CUDA" AND NOT VLLM_GPU_ARCHES)
    foreach(_ARCH IN LISTS CUDA_ARCHS)
        # 去掉小数点：8.6 -> 86。
        string(REPLACE "." "" _ARCH_NUM "${_ARCH}")
        # 去掉 "-real" / "+PTX" 等后缀，只保留架构数字。
        # 例：
        #   80+PTX  -> 80
        #   90-real -> 90
        string(REGEX REPLACE "[-+].*$" "" _ARCH_NUM "${_ARCH_NUM}")
        if(_ARCH_NUM)
            list(APPEND VLLM_GPU_ARCHES "${_ARCH_NUM}-real")
        endif()
    endforeach()
endif()

# 配置阶段打印最终传给 flash-attention 的目标 GPU 架构。
# 这类 message(STATUS ...) 不影响构建，只用于排查配置结果。
message(STATUS "flash-attention target architectures: ${VLLM_GPU_ARCHES}")

# 选择 flash-attention 源码来源：
#   - FLASH_ATTENTION_SRC_DIR 非空：使用本地源码；
#   - 否则：从 GitHub 下载指定 commit。
if(FLASH_ATTENTION_SRC_DIR)
    # FetchContent_Declare(SOURCE_DIR ...) 要求路径明确。
    # 如果用户传的是相对路径，这里转成绝对路径，避免后续外部项目内部
    # 改变当前目录时解析出错。
    if(NOT IS_ABSOLUTE FLASH_ATTENTION_SRC_DIR)
        get_filename_component(FLASH_ATTENTION_SRC_DIR
            "${FLASH_ATTENTION_SRC_DIR}" ABSOLUTE)
    endif()

    message(STATUS "Using local flash-attention source: ${FLASH_ATTENTION_SRC_DIR}")

    # 声明一个名为 flash_attention 的 FetchContent 依赖。
    #
    # SOURCE_DIR:
    #   使用本地源码目录，不下载。
    #
    # BINARY_DIR:
    #   外部项目自己的构建中间文件放到主构建目录下的 flash-attention 子目录：
    #       ${CMAKE_BINARY_DIR}/flash-attention
    #   避免与本项目目标的中间文件混在一起。
    FetchContent_Declare(
        flash_attention
        SOURCE_DIR ${FLASH_ATTENTION_SRC_DIR}
        BINARY_DIR ${CMAKE_BINARY_DIR}/flash-attention
    )
else()
    # 从 GitHub 拉取 flash-attention 源码。
    #
    # GIT_REPOSITORY:
    #   仓库地址。
    #
    # GIT_TAG:
    #   分支 / tag / commit。这里是固定 commit。
    #
    # GIT_PROGRESS TRUE:
    #   下载时打印 Git 进度，方便知道配置阶段是否卡在下载。
    #
    # BINARY_DIR:
    #   同上，指定外部项目构建目录。
    FetchContent_Declare(
        flash_attention
        GIT_REPOSITORY ${FLASH_ATTENTION_GIT_REPOSITORY}
        GIT_TAG ${FLASH_ATTENTION_GIT_TAG}
        GIT_PROGRESS TRUE
        BINARY_DIR ${CMAKE_BINARY_DIR}/flash-attention
    )
endif()

# 在执行 FetchContent_MakeAvailable 之前，临时设置全局输出目录。
#
# 为什么要在 MakeAvailable 之前设置：
#   FetchContent_MakeAvailable(flash_attention) 会执行外部项目的 CMakeLists.txt，
#   外部项目会在这个过程中 add_library(...) 创建自己的 target。
#
# CMAKE_LIBRARY_OUTPUT_DIRECTORY / CMAKE_RUNTIME_OUTPUT_DIRECTORY 是创建 target 时
# 会被 CMake 用作默认输出目录的变量。
#
# 这一步可以影响那些"没有主动设置自己输出目录"的 flash-attention target。
#
# 注意：
#   有些外部 target 可能自己设置了 LIBRARY_OUTPUT_DIRECTORY。
#   所以下面 MakeAvailable 之后还会递归收集 target，再用 set_target_properties
#   强制设置一次，作为兜底。
#
# 先保存旧值，是为了避免影响当前文件之后可能继续创建的其他目标。
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(_FLASH_ATTENTION_HAD_LIBRARY_OUTPUT_DIRECTORY TRUE)
    set(_FLASH_ATTENTION_OLD_LIBRARY_OUTPUT_DIRECTORY
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
endif()

if(DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(_FLASH_ATTENTION_HAD_RUNTIME_OUTPUT_DIRECTORY TRUE)
    set(_FLASH_ATTENTION_OLD_RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

# 临时把"接下来创建的 target"的默认产物输出目录设置为项目根目录。
#
# 这两行影响的是 CMake target 的默认输出属性初始化：
#
#   CMAKE_LIBRARY_OUTPUT_DIRECTORY
#       用来初始化后续 target 的 LIBRARY_OUTPUT_DIRECTORY。
#       主要影响：
#           - SHARED_LIBRARY 在 Linux 上生成的 .so；
#           - MODULE_LIBRARY 在 Linux 上生成的 .so；
#           - Python 扩展模块通常就是 MODULE_LIBRARY，
#             例如 xxx.cpython-313-x86_64-linux-gnu.so。
#
#   CMAKE_RUNTIME_OUTPUT_DIRECTORY
#       用来初始化后续 target 的 RUNTIME_OUTPUT_DIRECTORY。
#       主要影响：
#           - add_executable 生成的可执行文件；
#           - Windows 上 DLL 这类 runtime artifact。
#
# 为什么设置为 ${PROJECT_SOURCE_DIR}：
#   本项目 ROPE_cuda / validation 的产物也输出到项目根目录。
#   flash-attention 的 .so 放到同一位置后，Python 在项目根目录运行时更容易
#   import / dlopen / torch.ops.load_library，不需要额外改 sys.path 或拼 build 子目录路径。
#
# 为什么要在 FetchContent_MakeAvailable 之前设置：
#   MakeAvailable 会执行 flash-attention 的 CMakeLists.txt。
#   flash-attention 的 add_library(...) target 会在这期间被创建。
#   CMAKE_*_OUTPUT_DIRECTORY 只有在 target 创建时作为默认值生效，
#   所以必须在 target 创建之前设置。
#
# 这不是 BINARY_DIR：
#   BINARY_DIR 是外部项目的构建中间目录，例如：
#       ${CMAKE_BINARY_DIR}/flash-attention
#   CMAKE_LIBRARY_OUTPUT_DIRECTORY / CMAKE_RUNTIME_OUTPUT_DIRECTORY
#   控制的是最终 .so / executable 产物输出到哪里。
#
# 限制：
#   如果外部项目在自己的 target 上显式设置了 LIBRARY_OUTPUT_DIRECTORY，
#   那么这个全局默认值可能被覆盖。
#   因此 MakeAvailable 之后下面还会递归收集 target，并用
#   set_target_properties(...) 再强制设置一次，作为兜底。
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}")

# flash-attention 自己的 CMakeLists.txt 会调用：
#
#   find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule)
#
# 注意它找的是 CMake 的 `Python` package，不是本项目顶层已经找过的 `Python3`
# package。若不显式指定，CMake 可能重新选到系统 Python，例如：
#
#   /usr/bin/python3.10
#
# 这个系统 Python 不一定安装了 torch，于是 flash-attention 后续执行：
#
#   import torch; print(torch.utils.cmake_prefix_path)
#
# 会报：
#
#   ModuleNotFoundError: No module named 'torch'
#
# flash-attention 的 cmake/utils.cmake 注释也说明：
#   To customize which python gets found, set the `Python_EXECUTABLE` variable.
#
# 因此这里把父项目 find_package(Python3 ...) 已经确定的 venv Python
# 转交给 flash-attention 使用，确保它导入的是同一个环境里的 torch。
#
# 为什么要先清理 Python / _Python cache：
#   IDE reload 通常复用同一个 build 目录。如果之前 flash-attention 配置失败过，
#   CMakeCache.txt 里可能已经残留 FindPython 的内部结果，例如：
#
#       _Python_EXECUTABLE=/usr/bin/python3.10
#       _Python_INCLUDE_DIR=_Python_INCLUDE_DIR-NOTFOUND
#
#   即使后来把 Python_EXECUTABLE 改成 venv Python，FindPython 也可能继续复用
#   这些旧的内部 cache，出现：
#
#       Could NOT find Python ...
#       Unable to find python matching: <venv>/bin/python3
#
#   所以这里在调用 flash-attention 的 find_package(Python ...) 前，只清理
#   generic Python 包的 cache 变量：Python_* 和 _Python_*。
#   注意不清理 Python3_* / _Python3_*，因为那些是顶层 find_package(Python3 ...)
#   已经正确找到的结果。
get_cmake_property(_FLASH_ATTENTION_CACHE_VARS CACHE_VARIABLES)
foreach(_cache_var IN LISTS _FLASH_ATTENTION_CACHE_VARS)
    if(_cache_var MATCHES "^_?Python_")
        unset(${_cache_var} CACHE)
    endif()
endforeach()

# 用 CACHE + FORCE 的原因：
#   清理旧 cache 后，显式写入新的 Python_EXECUTABLE，确保 flash-attention 后续
#   find_package(Python ...) 以顶层 venv Python 为准。
if(Python3_EXECUTABLE)
    get_filename_component(_FLASH_ATTENTION_PYTHON_ROOT
        "${Python3_EXECUTABLE}" DIRECTORY)
    get_filename_component(_FLASH_ATTENTION_PYTHON_ROOT
        "${_FLASH_ATTENTION_PYTHON_ROOT}" DIRECTORY)
    set(Python_ROOT_DIR "${_FLASH_ATTENTION_PYTHON_ROOT}"
        CACHE PATH "Python root used by flash-attention" FORCE)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}"
        CACHE FILEPATH "Python executable used by flash-attention" FORCE)
    message(STATUS "flash-attention Python executable: ${Python_EXECUTABLE}")
endif()

# 真正让 FetchContent 生效。
#
# 这个命令会：
#   1. 如果源码不存在，则下载 / checkout flash-attention；
#   2. 准备源码目录和构建目录；
#   3. 调用 add_subdirectory，把 flash-attention 的 CMakeLists.txt 纳入当前构建。
#
# 执行之后，flash-attention 内部定义的 CMake target 才会出现在本项目构建图里。
#
# 对名为 flash_attention 的 FetchContent 内容，CMake 会生成变量：
#   flash_attention_SOURCE_DIR
#   flash_attention_BINARY_DIR
FetchContent_MakeAvailable(flash_attention)
message(STATUS "flash-attention is available at ${flash_attention_SOURCE_DIR}")

# 恢复 MakeAvailable 之前的全局输出目录设置。
#
# 如果原本有定义，就恢复原值；
# 如果原本没定义，就 unset，保持调用本文件之前的状态。
if(_FLASH_ATTENTION_HAD_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        "${_FLASH_ATTENTION_OLD_LIBRARY_OUTPUT_DIRECTORY}")
else()
    unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
endif()
if(_FLASH_ATTENTION_HAD_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        "${_FLASH_ATTENTION_OLD_RUNTIME_OUTPUT_DIRECTORY}")
else()
    unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
endif()

# 递归收集某个 CMake directory 及其子目录里定义的 target。
#
# 参数：
#   _dir:
#       要扫描的 CMake directory。这里传 flash_attention_SOURCE_DIR。
#
#   _out:
#       输出变量名。函数结束时用 PARENT_SCOPE 把结果写回调用者作用域。
#
# 用到的 CMake directory properties：
#   BUILDSYSTEM_TARGETS:
#       当前 directory 里 add_library / add_executable 创建的 target 列表。
#
#   SUBDIRECTORIES:
#       当前 directory 通过 add_subdirectory 加入的子 directory 列表。
#       注意这里说的是"CMake 子目录"，不是文件系统下所有子文件夹。
#       只有某个已执行的 CMakeLists.txt 里调用了 add_subdirectory(xxx)，
#       xxx 才会进入这个 SUBDIRECTORIES property。
#
#       例如 flash-attention 源码树下可能有：
#           csrc/
#           hopper/
#           flash_attn/
#           tests/
#       但这些目录只有在 CMakeLists.txt 中被 add_subdirectory(...) 加入后，
#       才会被这里遍历到。
#       如果 flash-attention 只是把 csrc/xxx.cpp、hopper/xxx.cu 当作 SOURCES
#       传给顶层 add_library / Python_add_library，而没有 add_subdirectory(csrc)
#       或 add_subdirectory(hopper)，它们就不是这里的 CMake 子目录。
#
# 为什么需要递归：
#   flash-attention 的 target 不一定都定义在顶层 CMakeLists.txt。
#   很多库目标可能在 csrc/、hopper/、flash_attn/ 等子目录中创建。
#   当前 vLLM flash-attention 版本主要在顶层 CMakeLists.txt 通过
#   define_gpu_extension_target(...) 创建 _vllm_fa2_C / _vllm_fa3_C；
#   但递归保留在这里，可以兼容其他版本或未来版本把 target 放进子目录CMakeLists.txt 的情况。
function(_flash_attention_collect_targets _dir _out)
    # 获取当前目录直接定义的 target。
    get_property(_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)

    # 获取当前目录通过 add_subdirectory(...) 加入的 CMake 子目录。
    #
    # 这里拿到的是 CMake 已经登记的 subdirectory 列表，不是对文件系统执行
    # find / glob。也就是说，它不会自动扫描 flash-attention 源码树下所有目录。
    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)

    # 深度优先遍历所有 CMake 子目录，把子目录里直接定义的 target 也合并进来。
    #
    # 举例：
    #   如果某个版本的 flash-attention 顶层 CMakeLists.txt 写了：
    #
    #       add_subdirectory(csrc)
    #
    #   而 csrc/CMakeLists.txt 里创建了：
    #
    #       add_library(flash_attn_core ...)
    #
    #   那么这一段递归会进入 csrc 这个 CMake directory，并把 flash_attn_core
    #   合并进 _targets。
    #
    #   如果没有 add_subdirectory(csrc)，即使文件系统中存在 csrc/，
    #   这里也不会遍历它。
    foreach(_subdir IN LISTS _subdirs)
        _flash_attention_collect_targets("${_subdir}" _subdir_targets)
        list(APPEND _targets ${_subdir_targets})
    endforeach()

    # CMake function 有自己的作用域。
    # PARENT_SCOPE 表示把结果写回调用者作用域，否则函数外拿不到 _targets。
    set(${_out} ${_targets} PARENT_SCOPE)
endfunction()

# 从 flash-attention 源码根目录开始，收集它创建出来的全部 target。
_flash_attention_collect_targets("${flash_attention_SOURCE_DIR}" _FLASH_ATTENTION_TARGETS)

# 遍历上面收集到的所有 flash-attention CMake target。
#
# _FLASH_ATTENTION_TARGETS 的来源：
#   - flash_attention_SOURCE_DIR 顶层 CMake directory 直接创建的 target；
#   - 以及它通过 add_subdirectory(...) 加入的 CMake 子目录中创建的 target。
#
# 这里不是遍历源码文件，也不是遍历文件系统目录，而是遍历 CMake target 名称。
# 当前 vLLM flash-attention 版本中，典型目标可能是：
#   _vllm_fa2_C
#   _vllm_fa3_C
#
# 本循环只处理会产出动态库 / Python 扩展 .so 的目标。
foreach(_target IN LISTS _FLASH_ATTENTION_TARGETS)
    # 查询 target 类型。
    #
    # 常见类型：
    #   STATIC_LIBRARY  静态库 .a
    #   SHARED_LIBRARY  共享库 .so
    #   MODULE_LIBRARY  Python 扩展 / dlopen 模块，通常也是 .so
    #   EXECUTABLE      可执行文件
    #   INTERFACE_LIBRARY 仅接口 target，无产物
    get_target_property(_target_type ${_target} TYPE)

    # 只关心动态可加载产物。
    #
    # 这些 target 才是我们希望输出到项目根目录的对象。
    # 对于 flash-attention，Python/CUDA 扩展通常是 MODULE_LIBRARY，
    # 普通动态库则是 SHARED_LIBRARY。
    #
    # 其他类型 target 的处理原则：
    #   STATIC_LIBRARY:
    #       静态库 .a，通常只是链接中间产物，不需要放项目根目录。
    #
    #   EXECUTABLE:
    #       可执行文件，不是本次需求关注的 flash-attention Python 扩展 .so。
    #
    #   INTERFACE_LIBRARY:
    #       没有实际产物，不能设置 .so 输出目录。
    #
    #   MODULE_LIBRARY:
    #       Python 扩展模块常见类型，比如 xxx.cpython-313-x86_64-linux-gnu.so。
    #
    #   SHARED_LIBRARY:
    #       普通共享库 .so，也可能被 Python 扩展或 torch op 加载。
    if(_target_type STREQUAL "MODULE_LIBRARY" OR
       _target_type STREQUAL "SHARED_LIBRARY")
        # 强制把该 target 的动态库输出目录设置到项目根目录。
        #
        # 为什么这里还要设置一次：
        #   前面的 CMAKE_LIBRARY_OUTPUT_DIRECTORY 只是 target 创建时的默认值。
        #   如果外部项目自己覆盖了 target 属性，默认值可能不起作用。
        #   这里在 target 已经创建之后直接 set_target_properties，
        #   优先级更明确。
        #
        # 为什么同时设置 DEBUG / RELEASE / RELWITHDEBINFO / MINSIZEREL：
        #   单配置生成器（Makefile / Ninja）主要用 LIBRARY_OUTPUT_DIRECTORY。
        #   多配置生成器（Ninja Multi-Config / Visual Studio）会按配置读取
        #   LIBRARY_OUTPUT_DIRECTORY_<CONFIG>。
        #   全部设置可以避免不同生成器行为不一致。
        set_target_properties(${_target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${PROJECT_SOURCE_DIR}"
        )

        # 记录被设置过输出目录的 target 名称，最后 message 打印出来。
        #
        # 这个列表可以用来确认第三个 foreach 实际筛选到了哪些目标。
        # 如果配置日志里这里为空，说明当前配置没有收集到 MODULE_LIBRARY /
        # SHARED_LIBRARY target，或者 flash-attention 配置阶段提前失败。
        list(APPEND _FLASH_ATTENTION_SO_TARGETS ${_target})
    endif()
endforeach()

# 打印最终被识别为共享库 / Python 扩展的 flash-attention target。
# 如果这里为空，说明：
#   1. flash-attention 当前配置没有创建 MODULE_LIBRARY / SHARED_LIBRARY；
#   2. 或者外部项目的 target 定义方式不在当前目录树属性里；
#   3. 或者配置条件不满足，相关 target 被外部项目跳过。
message(STATUS
    "flash-attention shared/module targets output to ${PROJECT_SOURCE_DIR}: "
    "${_FLASH_ATTENTION_SO_TARGETS}")
