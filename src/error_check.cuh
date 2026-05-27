#pragma once

#include <cuda_runtime.h>  // cudaError_t, cudaGetErrorString
#include <cstdio>          // fprintf, stderr
#include <cstdlib>         // exit, EXIT_FAILURE

// cudaCheck(cudaError_t, const char*, int)：底层实现，三个参数版本
//   error : CUDA API 返回的错误码（cudaSuccess 表示成功）
//   file  : 调用处的源文件名，类型 const char*
//             __FILE__ 是预处理器内置宏，展开为字符串字面量（如 "src/runner.cuh"）
//             字符串字面量类型为 const char[]，传参时自动退化为 const char*
//             与 std::string 的区别：
//               std::string  : 类对象，内部管理堆上的字符缓冲区，不会退化为 char*，需 .c_str() 转换
//               const char[] : 编译期常量，存储在只读数据段，传参退化为 const char*，零开销
//             此处用 const char* 而非 std::string，是因为 __FILE__ 是编译期常量，
//             无需运行时构造堆对象
//   line  : 调用处的行号，由宏 __LINE__ 在预处理阶段展开为整数
//   逻辑：若 error != cudaSuccess，打印文件名、行号、错误描述后终止程序
//   inline：此头文件会被多个 .cu 文件 #include，每个翻译单元各有一份函数定义。
//           inline 告知链接器这是合法的多份定义，自动合并，防止 ODR 重复定义错误。
//           （若不加 inline，链接时报 "multiple definition of cudaCheck" 错误）
//           性能方面：inline 同时建议编译器内联展开，消除函数调用开销，属于次要作用。
inline void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    // cudaGetErrorString：将错误码转为可读字符串，如 "invalid device pointer"
    fprintf(stderr,"[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

// 单参数宏：调用处只写 cudaCheck(expr)，宏自动填入 __FILE__ 和 __LINE__
//   __FILE__ / __LINE__ 在预处理阶段展开为调用处的文件名和行号，无运行时开销
//   外层括号：防止宏展开结果在复杂表达式中因运算符优先级被截断（防御性写法）
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))
