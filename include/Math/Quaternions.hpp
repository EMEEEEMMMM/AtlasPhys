#ifndef MATH_QUATERNIONS_HPP
#define MATH_QUATERNIONS_HPP

#include "Vector3.hpp"

namespace Math
{
    struct Quaternions
    {
        float w, x, y, z;
        Quaternions(float w = 0.0f, float x = 0.0f, float y = 0.0f, float z = 0.0f) : w(w), x(x), y(y), z(z) {}
        
        static Quaternions formAxisAngle(const Vector3& v, float theta) {
            float halfTheta = theta / 2;
            double s = sin(halfTheta);
            return Quaternions(cos(halfTheta), v.x * s, v.y * s, v.z * s);
        }

        Quaternions operator*(const Quaternions& p) const {
            return Quaternions(
                w * p.w - x * p.x - y * p.y - z * p.z,
                w * p.x + x * p.w + y * p.z - z * p.y,
                w * p.y - x * p.z + y * p.w - z * p.x,
                w * p.z + x * p.y - y * p.x + z * p.w
            );
        }

        void normalize() {
            double len = sqrt(w * w + x * x + y * y + z * z);
            if (len < 1e-6f) return;
            w /= len; x /= len; y /= len; z/= len;
        }
    };    
}


#endif
