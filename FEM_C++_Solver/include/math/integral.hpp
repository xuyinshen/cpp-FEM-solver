#pragma once

#include <functional>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "math/array.hpp"

namespace math
{
    double integrate_trapezoid(const std::function<double(double)>& f, double a, double b, long long n);
    double integrate_trapezoid(const std::vector<double>& f, const std::vector<double>& x);
    //double integrate_trapezoid(const Tensor f, const Matrix x, const Matrix y, const Matrix z);

    double integrate_simpson(const std::function<double(double)>& f, double a, double b, long long n);
    double integrate_simpson(const std::vector<double>& f, const std::vector<double>& x);
    //double integrate_simpson(const Matrix f, const Matrix x);

    
}