#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include "Math/Vector3.hpp"

#include <vector>
#include <cmath>
#include <memory>

namespace G_Objects { struct P_Objects; } 

namespace Math {
    extern float GRAVITY;
    inline float radians(float degrees) { return degrees * (M_PI / 180.0f); }
    inline float degrees(float radians) { return radians * (180.0f / M_PI); }

    inline Math::Vector3 GetVertsFromVertices(const std::vector<float>& verts, int i) {
        return Math::Vector3(verts[i * 7], verts[i * 7 + 1], verts[i * 7 + 2]);
    }

    void InitializePyramidAxes(std::shared_ptr<G_Objects::P_Objects> pyramid, int baseSides);
}

#endif