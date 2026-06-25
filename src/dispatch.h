#pragma once
// #pragma once 的等价写法 = 传统【文件级 include guard】(二选一，别叠加)：
//     #ifndef DISPATCH_H          // 用一个【全工程唯一】的宏名作“本文件已包含”标记
//     #define DISPATCH_H          // 首次包含：定义标记，继续往下编译文件内容
//       ... 文件全部内容 ...
//     #endif // DISPATCH_H        // 再次包含：DISPATCH_H 已定义→#ifndef 为假→整段跳过
//   二者作用相同(同一 TU 只展开一次本文件)；#pragma once 更简洁、无需起唯一宏名、
//   不怕宏名撞车，但非标准(主流编译器都支持)；include guard 是标准 C++、可移植性最佳。
//
// ── 关于本文件里的 #define 是否需要 #ifndef 保唯一 ───────────────────────────
//  · 防“同一翻译单元重复展开本文件” → 顶部的 #pragma once 已搞定，【无需】给每个宏套 #ifndef。
//  · #ifndef 宏名/#define/#endif 是另一种用途：“仅当未定义才定义”(可被外部覆盖的默认值，
//    如 CMakeLists 的 ROPE_VARIANT)。它【不保证】本份定义生效——若别处先定义了同名宏，会
//    静默跳过、采用别人的版本，所以它表达的是“先到先得”，不是“唯一”。
//  · 宏【无视命名空间】，是全 TU 文本替换 → 真正风险是【跨头文件同名冲突】。本文件的 EMPTY、
//    DISPATCH_CASE、DISPATCH_SWITCH、DISPATCH_FLOATING_TYPES 等名字较通用(且 torch/extension.h
//    在前，自带大量 AT_DISPATCH_*/DISPATCH_* 宏)，撞名时若定义不同会报 "macro redefinition"。
//  · 降冲突的正确做法不是 #ifndef，而是：①给这些宏加项目前缀(如 ROPE_*)；或②仅内部用时
//    用完 #undef。当前作内部 dispatch 用，留意上述命名风险即可。
// 详见 docs/c++/macro.txt。
// ─────────────────────────────────────────────────────────────────────────────


#include <torch/headeronly/util/Half.h>         // c10::Half        ← 轻量 headeronly 头
#include <torch/headeronly/util/BFloat16.h>     // c10::BFloat16    ← 取代重型 <torch/extension.h>
#include <string>                                // std::to_string（head_dim 报错信息）
#include <stdexcept>                             // std::runtime_error
#include <utility>                               // std::forward

// [[noreturn]] 抛出辅助：让编译器【确知】此调用不会返回 → switch 的 default 走它之后
//   控制流不可达，既无 -Wimplicit-fallthrough（穿透）顾虑，也不必靠 break 兜。
//   inline + 在头文件中定义：多 TU 包含不冲突（隐式 inline 的 ODR 豁免）。
namespace rope_dispatch {
template <typename Msg>
[[noreturn]] inline void throw_runtime(Msg&& msg) {
  throw std::runtime_error(std::forward<Msg>(msg));
}
}  // namespace rope_dispatch

#define ROPE_EMPTY(...)

#define ROPE_DISPATCH_CASE_TMPL(CASE_TYPE_USING_HINT, enum_type, ...)    \
  CASE_TYPE_USING_HINT(enum_type,scalar_t,__VA_ARGS__)

#define ROPE_PRIVATE_CASE_TYPE_USING_HINT_TMPL(PRELUDE, enum_type, HINT, ...)  \
    case enum_type: {                                                     \
    PRELUDE(enum_type);                                                   \
    /* enum_type 在 case 标签里是【编译期常量】(如 ScalarType::Half)，故可作    */ \
    /* 模板实参；ScalarTypeToCPPTypeT 把“枚举值”映射成对应 C++“类型”，          */ \
    /* 别名为 HINT(本宏链固定为 scalar_t)，供下方 lambda 体按名字引用。          */ \
    /* 这是 dispatch“运行时枚举值 → 编译期类型”的核心一步；运行时值(如          */ \
    /* qkv.scalar_type())不能直接做模板实参，必须先经 switch 落到某个 case。     */ \
    /* [[maybe_unused]]：lambda 体可能用不到 scalar_t，抑制 unused 警告。        */ \
    using HINT [[maybe_unused]] =                                         \
    torch::headeronly::impl::ScalarTypeToCPPTypeT<enum_type>;             \
     return __VA_ARGS__();                                                 \
    }

#define ROPE_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)  \
  ROPE_PRIVATE_CASE_TYPE_USING_HINT_TMPL(ROPE_EMPTY, enum_type, HINT, __VA_ARGS__)

#define ROPE_DISPATCH_CASE(enum_type, ...)    \
  ROPE_DISPATCH_CASE_TMPL(ROPE_PRIVATE_CASE_TYPE_USING_HINT,enum_type, __VA_ARGS__)

#define ROPE_DISPATCH_CASE_FLOATING_TYPES(...)    \
  ROPE_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)   \
  ROPE_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)  \
  ROPE_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)


#define ROPE_DISPATCH_SWITCH_TMPL(PRELUDE, CHECK_NOT_IMPLEMENTED, TYPE, NAME, ...)   \
  [&] {                                                                     \
  const auto& the_type = TYPE;                                            \
  constexpr const char* at_dispatch_name = NAME;                          \
  PRELUDE(at_dispatch_name, the_type);                                         \
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")             \
  switch (the_type) {                                                          \
  __VA_ARGS__                                                           \
  default:                                                              \
  CHECK_NOT_IMPLEMENTED(                                              \
  false,                                                              \
  '"',                                                                \
  at_dispatch_name,                                                   \
  "\" not implemented for '",                                         \
  torch::headeronly::toString(the_type),                               \
  "'");                                                               \
  }                                                                       \
  C10_DIAGNOSTIC_POP()                                                    \
  }()


#define ROPE_ST_TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {           \
  rope_dispatch::throw_runtime(STD_TORCH_CHECK_MSG( \
  cond,                                     \
  "",                                       \
  __func__,                                 \
  ", ",                                     \
  __FILE__,                                 \
  ":",                                      \
  __LINE__,                                 \
  ", ",                                     \
  ##__VA_ARGS__));                          \
  }


#define ROPE_DISPATCH_SWITCH(TYPE, NAME, ...) \
  ROPE_DISPATCH_SWITCH_TMPL(ROPE_EMPTY, ROPE_ST_TORCH_CHECK, TYPE, NAME, __VA_ARGS__)

#define ROPE_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)    \
  ROPE_DISPATCH_SWITCH(TYPE, NAME,ROPE_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))


#define ROPE_DISPATCH_HEAD_DIM(TYPE,HEAD_DIM,NAME,...)    \
  ROPE_DISPATCH_HEAD_DIM_TMPL(ROPE_ST_TORCH_CHECK,TYPE,HEAD_DIM,NAME,__VA_ARGS__)


#define ROPE_DISPATCH_HEAD_DIM_TMPL(CHECK_NOT_IMPLEMENTED,TYPE,HEAD_DIM,NAME,...)    \
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")                        \
  switch (TYPE) {                                                                    \
    case torch::headeronly::ScalarType::Float: {                                      \
      switch (HEAD_DIM) {                                                               \
        case 32:{                                                                     \
         constexpr uint headDIM = 32;                                                  \
         return __VA_ARGS__();                                                         \
        }                                                                             \
        case 64:{                                                                      \
        constexpr uint headDIM = 64;                                                  \
        return __VA_ARGS__();                                                         \
        }                                                                             \
        case 128:{                                                                      \
        constexpr uint headDIM = 128;                                                  \
        return __VA_ARGS__();                                                         \
        }                                                                             \
        default:                                                                     \
        CHECK_NOT_IMPLEMENTED(                                                      \
        false,                                                                      \
        '"',                                                                        \
        NAME,                                                                      \
        "\" not implemented for HEAD_DIM '",                                        \
        std::to_string(HEAD_DIM),                                                     \
        "'");                                                                        \
      }                                                                               \
    }                                                                              \
    case torch::headeronly::ScalarType::Half: {                                        \
      switch (HEAD_DIM) {                                                               \
        case 64:{                                                                     \
         constexpr uint headDIM = 64;                                                  \
         return __VA_ARGS__();                                                         \
        }                                                                             \
        case 128:{                                                                      \
        constexpr uint headDIM = 128;                                                  \
        return __VA_ARGS__();                                                         \
        }                                                                             \
        case 256:{                                                                      \
        constexpr uint headDIM = 256;                                                  \
        return __VA_ARGS__();                                                         \
        }                                                                             \
        default:                                                                     \
        CHECK_NOT_IMPLEMENTED(                                                      \
        false,                                                                      \
        '"',                                                                        \
        NAME,                                                                      \
        "\" not implemented for HEAD_DIM'",                                         \
        std::to_string(HEAD_DIM),                                                     \
        "'");                                                                        \
      }                                                                               \
    }                                                                                 \
    case torch::headeronly::ScalarType::BFloat16: {                                    \
      switch (HEAD_DIM) {                                                               \
        case 64:{                                                                     \
         constexpr uint headDIM = 64;                                                  \
         return __VA_ARGS__();                                                         \
        }                                                                             \
        case 128:{                                                                      \
        constexpr uint headDIM = 128;                                                  \
        return __VA_ARGS__();                                                         \
        }                                                                             \
        case 256:{                                                                      \
        constexpr uint headDIM = 256;                                                  \
        return __VA_ARGS__();                                                         \
        }                                                                             \
        default:                                                                     \
        CHECK_NOT_IMPLEMENTED(                                                      \
        false,                                                                      \
        '"',                                                                        \
        NAME,                                                                      \
        "\" not implemented for HEAD_DIM '",                                        \
        std::to_string(HEAD_DIM),                                                     \
        "'");                                                                        \
      }                                                                              \
    }                                                                                 \
    default:                                                                          \
      CHECK_NOT_IMPLEMENTED(                                                          \
      false,                                                                          \
      '"',                                                                            \
      NAME,                                                                           \
      "\" not implemented for dtype '",                                               \
      torch::headeronly::toString(TYPE),                                              \
      "'");                                                                           \
  }                                                                                   \
  C10_DIAGNOSTIC_POP()

