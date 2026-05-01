#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <cmath>

namespace Math {
    extern float GRAVITY;
    inline float radians(float degrees) { return degrees * (M_PI / 180.0f); }
    inline float degrees(float radians) { return radians * (180.0f / M_PI); }
}

#endif