#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>     // 用于高精度计时

#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>      // OpenMP 运行时 API
#endif

#include "math/integral.hpp"

int main()
{
    const double a = 0.0;
    const double b = 1.0;
    // 测试用：10亿次。如果要跑10万亿请做好运行几小时的心理准备
    const long long n = 1000000000; 

    // 提取成变量，避免每次传参都重新构造 std::function
    auto f1 = [](double x) { return std::sin(x); };

    std::cout << std::fixed << std::setprecision(10);

#ifdef CPPNUM_HAVE_OPENMP

    const int max_threads = omp_get_max_threads();
    std::cout << "[info] 可用线程数 = " << max_threads
              << " (物理/逻辑核心数 = " << omp_get_num_procs() << ")\n\n";

    // ================= 1. 强制串行测试 (1个线程) =================
    omp_set_num_threads(1); 
    
    auto start_serial = std::chrono::high_resolution_clock::now();
    const double r_serial = math::integrate_simpson(f1, a, b, n) - math::integrate_trapezoid(f1, a, b, n);
    auto end_serial = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> time_serial = end_serial - start_serial;

    // ================= 2. 并行测试 (所有可用线程) =================
    omp_set_num_threads(max_threads); 
    
    auto start_parallel = std::chrono::high_resolution_clock::now();
    const double r_parallel = math::integrate_simpson(f1, a, b, n) - math::integrate_trapezoid(f1, a, b, n);
    auto end_parallel = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> time_parallel = end_parallel - start_parallel;

    // ================= 3. 输出对比报告 =================
    std::cout << "=== 积分结果验证 ===\n";
    std::cout << "Serial result   : " << r_serial << "\n";
    std::cout << "Parallel result : " << r_parallel << "\n";
    std::cout << "Exact value     : " << (1.0 - std::cos(1.0)) << "\n\n";

    std::cout << "=== 性能对比 (n = " << n << ") ===\n";
    std::cout << "Serial time   (" << 1 << " thread ) : " << time_serial.count() << " s\n";
    std::cout << "Parallel time (" << max_threads << " threads) : " << time_parallel.count() << " s\n";
    
    double speedup = time_serial.count() / time_parallel.count();
    std::cout << "Speedup (加速比) : " << speedup << "x\n";

#else
    // 如果编译时没开 OpenMP，就只跑一次纯串行
    auto start = std::chrono::high_resolution_clock::now();
    const double r = math::integrate_simpson(f1, a, b, n) - math::integrate_trapezoid(f1, a, b, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_elapsed = end - start;

    std::cout << "Result (Serial Build) : " << r << "\n";
    std::cout << "Time                  : " << time_elapsed.count() << " s\n";
#endif

    return 0;
}