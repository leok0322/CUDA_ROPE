#include "error_check.cuh"  // cudaCheck 宏：包装 CUDA API 调用，失败时打印文件名/行号并终止
#include "runner.cuh"        // run_softmax_kernel_base / run_softmax_kernel_naive 等 kernel 启动函数

#include <cuda_runtime.h>   // cudaMalloc/cudaFree/cudaMemcpy/cudaEvent_t/cudaSetDevice 等 CUDA Runtime API

#include <string>           // std::string：errLogFile / resultDir / resultLogFile 字符串变量
#include <vector>           // std::vector<int>：SIZE 存放各测试规模
#include <iostream>         // std::cout / std::cerr / std::endl：控制台输出
#include <fstream>          // std::ofstream：写入结果日志文件和错误日志文件
#include <filesystem>       // std::filesystem::create_directories：自动创建 benchmark_results 目录（C++17）
#include <iomanip>          // std::setprecision / std::setw / std::fixed：控制浮点输出格式
#include <cstdio>           // fprintf / fflush / stdout / stderr：C 风格格式化输出
#include <cstdlib>          // malloc / free / exit / getenv / std::atoi / srand / rand：内存与进程控制
#include <cstring>          // memset：将 out 缓冲区初始化为 0
#include <cmath>            // std::fabs / isnan：验证矩阵时计算绝对误差
#include <sys/time.h>       // timeval / gettimeofday：获取微秒级时间戳，用于随机数种子
#include <stdexcept>        // std::invalid_argument：run_kernel default 分支抛出非法参数异常
#include <algorithm>        // std::max：取两个同类型值的较大值；std::fill：将区间填充为指定值


const std::string errLogFile = "matrixValidationFailure.txt";
const std::string resultDir  = "benchmark_results";

// randomize_matrix：用随机浮点数填充矩阵，范围约 [-5, 5]
//   以当前微秒时间戳作为随机种子，避免每次运行产生相同序列
//   mat：CPU 内存指针；N：元素总数
void randomize_matrix(float *mat, uint N) {
    timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.f);
        mat[i] = tmp;
    }
}

// verify_matrix：逐元素对比 kernel 输出（matOut）与参考结果（matRef）
//   允许误差 0.01（浮点计算精度差异）
//   发现误差超限或 NaN 时打印错误信息并返回 false
bool verify_matrix(const float *matRef, const float *matOut, uint64_t N) {
    double diff = 0.0;
    for (int i = 0; i < N; i++) {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (isnan(diff) || diff > 0.01) {
            fprintf(stderr, "Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                    matRef[i], matOut[i], diff, i);
            return false;
        }
    }
    return true;
}

// print_matrix：将矩阵内容以可读格式写入文件流，用于验证失败时记录输入/输出
//   M×N 矩阵，每行 N 个元素，行间以 ";\n" 分隔，整体用 [] 包裹
void print_matrix(const float *A, uint M, uint N, std::ofstream &fs) {
    fs << std::setprecision(2) << std::fixed;
    fs << "[";
    for (int i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            fs << std::setw(5) << A[i];
        else
            fs << std::setw(5) << A[i] << ", ";
        if ((i + 1) % N == 0 && i + 1 < M * N)
            fs << ";\n";
    }
    fs << "]\n";
}

// run_kernel：按 kernel_num 分发到对应的 softmax kernel 实现
//   参数默认值只需在头文件的函数声明中写，不在定义处重复
//   deviceIdx：目标 GPU 设备编号（多卡时使用）
void run_kernel(const int kernel_num, const uint totalRow, const uint totalCol, float *A,
                float *out) {
    switch (kernel_num) {
        case 0:  run_ROPE_kernel_base(totalRow, totalCol, A, out);  break;
        default:
            // throw：抛出异常，沿调用栈向上传播直到被 catch 捕获
            // std::invalid_argument：标准异常类，表示传入参数不合法
            // main() 入口已校验 kernel_num 范围（0-12），正常流程不会走到这里
            // 此处作为防御性编程的最后保障，防止被其他地方以非法参数调用
            throw std::invalid_argument("Unknown kernel number");
    }
}

int main(int argc, char **argv) {
    // ── 参数解析 ──────────────────────────────────────────────────────────────
    if (argc != 2) {
        std::cerr << "Please select a kernel (range 0 - 12, 0 for base reference)"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > 10) {
        std::cerr << "Please enter a valid kernel number (0-9)" << std::endl;
        exit(EXIT_FAILURE);
    }

    // ── 设备选择 ──────────────────────────────────────────────────────────────
    // DEVICE 环境变量：多卡机器上指定使用哪张 GPU（如 DEVICE=1 ./validation 1）
    int deviceIdx = 0;
    if (getenv("DEVICE") != nullptr)
        deviceIdx = std::atoi(getenv("DEVICE"));
    cudaCheck(cudaSetDevice(deviceIdx));
    fprintf(stdout, "Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    // ── CUDA 计时事件 ─────────────────────────────────────────────────────────
    // cudaEvent_t：GPU 侧时间戳，cudaEventRecord 将事件插入 GPU 命令流
    // cudaEventElapsedTime：计算两个事件之间的 GPU 执行时间（毫秒）
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    // ── 测试规模 ──────────────────────────────────────────────────────────────
    // 对每种 size，矩阵形状为 [msize, nsize]
    std::vector<uint> mSIZE = {128, 256, 512, 1024, 2048, 4096};
    std::vector<uint> nSIZE = {128, 256, 512, 1024, 2048, 4096};
    uint m, n, max_size;
    max_size = std::max(mSIZE[mSIZE.size() - 1],nSIZE[nSIZE.size() - 1]);   // 按最大规模分配内存，复用同一块缓冲区
    std::cout << "Max size: " << max_size << std::endl;

    // ── CPU 内存分配 ──────────────────────────────────────────────────────────
    // A      : 输入矩阵（随机初始化）
    // out    : 待测 kernel 的输出缓冲区（初始化为 0，避免残留随机值干扰验证）
    // out_ref: 参考 kernel（base）的输出缓冲区（初始化为 0）
    float *A = nullptr, *out = nullptr, *out_ref = nullptr;
    A       = static_cast<float *>(malloc(sizeof(float) * max_size * max_size));
    out     = static_cast<float *>(malloc(sizeof(float) * max_size * max_size));
    out_ref = static_cast<float *>(malloc(sizeof(float) * max_size * max_size));

    // 输入矩阵随机填充，输出缓冲区清零
    // out 初始化为 0 而非随机数：kernel 应将每个元素完整写出，
    //   若有元素未被写入，值为 0 比随机数更容易在验证时被检测到
    randomize_matrix(A, max_size * max_size);
    memset(out,     0, sizeof(float) * max_size * max_size);
    memset(out_ref, 0, sizeof(float) * max_size * max_size);

    // ── GPU 内存分配与数据传输 ────────────────────────────────────────────────
    float *dA = nullptr, *dout = nullptr, *dout_ref = nullptr;
    // cudaMalloc 签名：cudaError_t cudaMalloc(void** devPtr, size_t size)
    //   参数类型为 void**，但 dA 是 float*，&dA 是 float**
    //   float** → void**：不存在隐式转换，static_cast 也不允许（无关联指针类型）
    //   必须用 reinterpret_cast：强制重新解释内存，告知编译器"底层布局兼容"
    //
    //   对比 void* ↔ float* 的规则（一级指针，有特殊待遇）：
    //     float* → void*：隐式转换，任何指针都可直接赋值给 void*，无需 cast
    //     void*  → float*：需 static_cast<float*>，程序员显式承诺类型正确
    //   void** 是"指向 void* 的指针"，没有上述特殊规则，与 float** 完全无关联
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dA),      sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dout),    sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc(reinterpret_cast<void **>(&dout_ref),sizeof(float) * max_size * max_size));


    // Host → Device：将 CPU 数据拷贝到 GPU 显存
    cudaCheck(cudaMemcpy(dA,       A,    sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dout,     out,  sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    // ── 主测试循环 ────────────────────────────────────────────────────────────
    long repeat_times {100};   // 每个 size 重复 50 次取平均，减少 GPU 调度抖动影响

    for (uint msize : mSIZE) {
        for (uint nsize : nSIZE) {
            m = msize;
            n = nsize;
            std::cout << "dimensions(m, n) " << m << ", "<< n <<std::endl;

            // ── 正确性验证（kernel 0 为参考基准）────────────────────────────────
            if (kernel_num != 0) {
                try {
                    run_kernel(0,          m, n, dA, dout_ref); // 参考 kernel（base）
                    run_kernel(kernel_num, m, n, dA, dout); // 待测 kernel
                    cudaCheck(cudaDeviceSynchronize()); // 等待 GPU 完成，确保结果就绪再拷回
                } catch (const std::exception &e) {
                    fprintf(stderr, "%s\n", e.what());
                    exit(EXIT_FAILURE);
                }

                // Device → Host：将 GPU 结果拷回 CPU 做数值对比
                cudaCheck(cudaMemcpy(out_ref, dout_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(out,     dout,     sizeof(float) * m * n, cudaMemcpyDeviceToHost));

                if (!verify_matrix(out_ref, out, static_cast<uint64_t>(m) * n)) {
                    std::cout << "Failed to pass the correctness verification." << std::endl;
                    if (m <= 128 && n <= 128) {
                        // 仅在小矩阵时记录完整数据，大矩阵文件过大
                        std::cout << " Logging faulty output into " << errLogFile << "\n";
                        std::ofstream fs;
                        // open() 默认 std::ios::out，隐含 std::ios::trunc：
                        //   清空文件内容，写指针归零 → 覆盖模式
                        //   "覆盖 vs 追加" 只在 open() 时生效，后续写入与此无关
                        fs.open(errLogFile);
                        // 以下三组写入是对同一个已打开流的顺序写入，不是"追加模式"：
                        //   每次 << 或 print_matrix 写完后写指针自动后移
                        //   文件只 open() 一次，写指针连续推进，不会相互覆盖
                        //   类比：打开文档清空内容后连续打三段话，每段接在上一段之后
                        fs << "A:\n";       print_matrix(A,       m, n, fs);
                        fs << "out:\n";     print_matrix(out,     m, n, fs);
                        fs << "out_ref:\n"; print_matrix(out_ref, m, n, fs);
                        // fs 在此作用域结束时析构，析构函数自动调用 close()
                    }
                    exit(EXIT_FAILURE);
                }
            }

            // ── 性能测试（repeat_times 次取平均）────────────────────────────────
            cudaEventRecord(beg);
            for (int j = 0; j < repeat_times; j++) {
                try {
                    run_kernel(kernel_num, m, n, dA, dout);
                } catch (const std::exception &e) {
                    fprintf(stderr, "%s\n", e.what());
                }
            }
            cudaEventRecord(end);
            cudaCheck(cudaEventSynchronize(end)); // 等待 end 事件完成，确保计时准确
            cudaEventElapsedTime(&elapsed_time, beg, end);
            elapsed_time /= 1000.f;  // 毫秒 → 秒

            // ── FLOPS 计算 ────────────────────────────────────────────────────────
            // RoPE 每个维度对（2 个元素）的旋转公式：
            //   out1 = x1*cos - x2*sin   → 2 mul + 1 sub = 3 FLOP
            //   out2 = x1*sin + x2*cos   → 2 mul + 1 add = 3 FLOP
            //   小计：6 FLOP / 维度对
            //
            // 每行（token）有 coupleNum = n/2 个维度对，共 m 行：
            //   总 FLOP = 6 × (n/2) × m = 3 × n × m
            //
            // 注：sin/cos/powf 等超越函数未计入（硬件以查表+迭代实现，
            //     FLOP 数不固定，benchmark 惯例只统计算术运算）
            long long floatPointOperations = static_cast<long long>(3 * n) * m;
            // 用 double 避免 float 精度不足（float 只有 ~7 位有效数字，大矩阵时丢失精度）
            double flops = (static_cast<double>(repeat_times) * static_cast<double>(floatPointOperations) * 1e-9) / elapsed_time;
            fprintf(stdout,
                    "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: (%u, %u).\n",
                    elapsed_time / static_cast<float>(repeat_times), flops, m, n);
            fflush(stdout);

            // ── 结果写入文件 ──────────────────────────────────────────────────────
            std::filesystem::create_directories(resultDir);
            const std::string resultLogFile = resultDir + "/ROPE_kernel_" + argv[1] + "_result.txt";
            // fs 在 for 循环体内声明：每次迭代创建新对象，迭代结束析构 → 自动 close()
            //   因此每次迭代都要重新 open()，open() 的模式决定文件内容是否保留
            std::ofstream fs;
            if (m == mSIZE[0] && n == nSIZE[0]) {
                // 第一个 size：覆盖模式（默认 std::ios::out | std::ios::trunc）
                //   清空上次运行的旧结果，写入本次运行的表头
                fs.open(resultLogFile);
                fs << "Running kernel " << kernel_num << " on device " << deviceIdx << ".\n";
            } else {
                // 后续 size：必须用 std::ios::app（追加模式）
                //   若用默认 open()，文件被 trunc 清空，前几次 size 的结果全部丢失
                //   std::ios::app：每次 open() 时写指针跳到文件末尾，保留已有内容
                fs.open(resultLogFile, std::ios::app);
            }
            // 以下写入是对同一个已打开流的顺序写入（写指针自动后移），与覆盖/追加模式无关
            fs << "dimensions(m, n) " << m << ", " << n <<"\n";
            fs << std::fixed << std::setprecision(6)
               << "Average elapsed time: (" << elapsed_time / static_cast<float>(repeat_times) << ") s, performance: (";
            fs << std::setprecision(1) << flops << ") GFLOPS. size: (" << m << ", " << n << ").\n";

        }

    }

    // ── 资源释放 ──────────────────────────────────────────────────────────────
    free(A);
    free(out);
    free(out_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dout));
    cudaCheck(cudaFree(dout_ref));
    cudaCheck(cudaEventDestroy(beg));
    cudaCheck(cudaEventDestroy(end));

    return 0;
}
