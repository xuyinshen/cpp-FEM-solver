#include <vector>
#include <cmath>
#include <functional>
#include <iostream>

#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>      // OpenMP 运行时 API
#endif

#include "math/integral.hpp"
#include "math/array.hpp"

int main() 
{
    std::vector<double> x = math::arange(0.0, 10.0, 0.0001);

    std::vector<double> f(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        f[i] = x[i] * x[i];
    }

    const int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads); 

    double result_simpson = math::integrate_simpson(f, x);
    double result_trapezoid = math::integrate_trapezoid(f, x);
    std::cout << "辛普森积分为" << result_simpson << std::endl;
    std::cout << "梯形积分为" << result_trapezoid << std::endl;

    
    return 0;
}