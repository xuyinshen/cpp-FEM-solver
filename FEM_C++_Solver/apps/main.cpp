#include <cmath>      // std::sin, std::cos
#include <iomanip>    // std::setprecision
#include <iostream>   // std::cout

#include "math/integral.hpp"   // 只需要声明，定义在库里

int main()
{
    const double a = 0.0;      // 积分下限
    const double b = 1.0;      // 积分上限
    const int    n = 100000000;   // 子区间数

    // 1) 传函数指针
    const double r1 = math::integrate_trapezoid(
        [](double x) { return std::sin(x); }, a, b, n);

    // 2) 传无捕获 lambda（可隐式转换成函数指针，同样合法）
    const double r2 = math::integrate_trapezoid(
        [](double x) { return x * x; }, a, b, n);

    std::cout << std::setprecision(10);
    std::cout << "int_0^1 sin(x) dx = " << r1
              << "  (精确值 " << 1.0 - std::cos(1.0) << ")\n";
    std::cout << "int_0^1 x^2    dx = " << r2
              << "  (精确值 " << 1.0 / 3.0 << ")\n";

    return 0;
}