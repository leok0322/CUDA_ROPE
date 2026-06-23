#pragma once

#include <torch/extension.h>
#include <torch/headeronly/util/Exception.h>

#define EMPTY(...)

#define DISPATCH_CASE_TMPL(CASE_TYPE_USING_HINT, enum_type, ...)    \
  CASE_TYPE_USING_HINT(enum_type,scalar_t,__VA_ARGS__)

#define PRIVATE_CASE_TYPE_USING_HINT_TMPL(PRELUDE, enum_type, HINT, ...)  \
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

#define PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)  \
  PRIVATE_CASE_TYPE_USING_HINT_TMPL(EMPTY, enum_type, HINT, __VA_ARGS__)

#define DISPATCH_CASE(enum_type, ...)    \
  DISPATCH_CASE_TMPL(PRIVATE_CASE_TYPE_USING_HINT,enum_type, __VA_ARGS__)

#define DISPATCH_CASE_FLOATING_TYPES(...)    \
  DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__)   \
  DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)  \
  DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)


#define DISPATCH_SWITCH_TMPL(PRELUDE, CHECK_NOT_IMPLEMENTED, TYPE, NAME, ...)   \
  [&] {                                                                     \
  const auto& the_type = TYPE;                                            \
  constexpr const char* at_dispatch_name = NAME;                          \
  PRELUDE(at_dispatch_name, the_type);                                         \
  C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")             \
  switch (the_type) {                                                          \
  __VA_ARGS__                                                           \
  default:                                                              \
  CHECK_NOT_IMPLEMENTED(                                              \
  false,                                                          \
  '"',                                                            \
  at_dispatch_name,                                               \
  "\" not implemented for '",                                     \
  torch::headeronly::toString(the_type),                               \
  "'");                                                           \
  }                                                                       \
  C10_DIAGNOSTIC_POP()                                                    \
  }()


#define ST_TORCH_CHECK(cond, ...)                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {           \
  throw std::runtime_error(STD_TORCH_CHECK_MSG( \
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


#define DISPATCH_SWITCH(TYPE, NAME, ...) \
  DISPATCH_SWITCH_TMPL(EMPTY, ST_TORCH_CHECK, TYPE, NAME, __VA_ARGS__)

#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)    \
  DISPATCH_SWITCH(TYPE, NAME,DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
