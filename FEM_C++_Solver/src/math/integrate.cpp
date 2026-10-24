#ifdef CPPNUM_HAVE_OPENMP
#include <omp.h>
#endif

#include "math/integral.hpp"
#include "math/array.hpp"
#include <iostream>

namespace math
{
    // =============== 梯形积分法 ====================

    double integrate_trapezoid(const std::function<double(double)>& f, double a, double b, long long n)
    {
        if (a > b)
        {
            std::cout << "积分上底小于积分下底！\n";
            return 0.0;
        }
        if (a == b)
        {
            return 0.0;
        }
        if (n < 1)
        {
            std::cout << "积分分割数小于 1！\n";
            return 0.0;
        }

        const double h = (b - a) / n;
        double total = 0.0;

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:total) schedule(static)
        #endif
        for (int i = 0; i < n; i++)
        {
            total += h * f(a + static_cast<double>(i)*h);
        }

        return total;
    }
    double integrate_trapezoid(const std::vector<double>& f, const std::vector<double>& x)
    {
        const size_t len_f = f.size();
        const size_t len_x = f.size();
        if (len_f != len_x)
        {
            std::cout << "被积向量和坐标向量的长度不一致！\n";
            return 0.0;
        }
        if (len_x < 2)
        {
            std::cout << "坐标向量的长度为 1 ！\n";
            return 0.0;
        }

        double total = 0.0;

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:total) schedule(static)
        #endif
        for (int i = 1; i < static_cast<int>(len_x); i++)
        {
            total += (f[i] + f[i-1]) * (x[i] - x[i-1]) * 0.5;
        }

        return total;
    }


    // ==================== 辛普森积分法 =========================

    double integrate_simpson(const std::function<double(double)>& f, double a, double b, long long n)
    {
        if (a > b)
        {
            std::cout << "积分上底小于积分下底！\n";
            return 0.0;
        }
        if (a == b)
        {
            return 0.0;
        }
        if (n < 1)
        {
            std::cout << "积分分割数小于 1！\n";
            return 0.0;
        }

        const double h = (b - a) / n;
        double total = 0.0;

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:total) schedule(static)
        #endif
        for (int i = 0; i < n; i++)
        {
            total += h / 6.0 * ( f(a + static_cast<double>(i)*h) + f(a + static_cast<double>(i)*h + h) + 4.0 * f(a + static_cast<double>(i)*h + 0.5*h) );
        }

        return total;
    }
    double integrate_simpson(const std::vector<double>& f, const std::vector<double>& x)
    {
        const size_t len_f = f.size();
        const size_t len_x = f.size();
        if (len_f != len_x)
        {
            std::cout << "被积向量和坐标向量的长度不一致！\n";
            return 0.0;
        }
        if (len_x < 3)
        {
            std::cout << "坐标向量的长度不够 ！\n";
            return 0.0;
        }

        double total = 0.0;

        #ifdef CPPNUM_HAVE_OPENMP
            #pragma omp parallel for reduction(+:total) schedule(static)
        #endif
        for (int i = 0; i < static_cast<int>(len_x/2); i++)
        {
            total += (x[2*i+2] - x[2*i]) / 6.0 * ( f[2*i] + f[2*i+2] + 4.0 * f[2*i+1] );
        } 

        return (total + ((static_cast<int>(len_x) % 2 == 1) ? 0.0 : (f[len_x-1] + f[len_x-2]) * (x[1] - x[0]) * 0.5)) ;
    }
}

