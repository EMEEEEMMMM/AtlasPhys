#ifndef MATH_MATRIX4_H
#define MATH_MATRIX4_H

#include <cmath>
#include <cstring>

#include "Vector3.hpp"
#include "Utils.hpp"

namespace Math {
struct Matrix4 {
    float m[4][4];

    Matrix4() { setIdentity(1.0f); }
    Matrix4(float v) { setIdentity(v); }
    Matrix4(float v1, float v2, float v3, float v4) { setEachIdentityDiag(v1, v2, v3, v4); }
    Matrix4(
        float v00, float v01, float v02, float v03,
        float v10, float v11, float v12, float v13,
        float v20, float v21, float v22, float v23,
        float v30, float v31, float v32, float v33
    ) { setEachIdentity(
        v00, v01, v02, v03,
        v10, v11, v12, v13,
        v20, v21, v22, v23,
        v30, v31, v32, v33
    ); }

    void setIdentity(float v) {
        std::memset(m, 0, sizeof(m));
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = v;
    }

    void setEachIdentityDiag(float v1, float v2, float v3, float v4) {
        std::memset(m, 0, sizeof(m));
        m[0][0] = v1;
        m[1][1] = v2;
        m[2][2] = v3;
        m[3][3] = v4;
    }

    void setEachIdentity(
        float v00, float v01, float v02, float v03,
        float v10, float v11, float v12, float v13,
        float v20, float v21, float v22, float v23,
        float v30, float v31, float v32, float v33
    ) {
        std::memset(m, 0, sizeof(m));
        m[0][0] = v00; m[0][1] = v01; m[0][2] = v02; m[0][3] = v03;
        m[1][0] = v10; m[1][1] = v11; m[1][2] = v12; m[1][3] = v13;
        m[2][0] = v20; m[2][1] = v21; m[2][2] = v22; m[2][3] = v23;
        m[3][0] = v30; m[3][1] = v31; m[3][2] = v32; m[3][3] = v33;
    }

    inline Matrix4 operator*(const Matrix4& matrix) const {
        Matrix4 resultMatrix(0.0f);
        for (int i = 0; i <= 3; i++) {
            for (int j = 0; j <= 3; j++) {
                for (int k = 0; k <= 3; k++) {
                    resultMatrix.m[i][j] += m[i][k] * matrix.m[k][j];
                }
            }
        }
        return resultMatrix;
    }

    static Matrix4 Ortho(float left, float right, float bottom, float top,
                         float near, float far) {
        Matrix4 matrix;
        matrix.m[0][0] = 2.0f / (right - left);
        matrix.m[1][1] = 2.0f / (top - bottom);
        matrix.m[2][2] = -2.0f / (far - near);
        matrix.m[3][0] = -(right + left) / (right - left);
        matrix.m[3][1] = -(top + bottom) / (top - bottom);
        matrix.m[3][2] = -(far + near) / (far - near);
        return matrix;
    }

    static Matrix4 Persp(float fovRad, float aspect, float near, float far) {
        Matrix4 matrix;
        std::memset(matrix.m, 0, sizeof(matrix.m));
        float tanHalfFov = std::tan(fovRad / 2.0f);
        matrix.m[0][0] = 1.0f / (aspect * tanHalfFov);
        matrix.m[1][1] = 1.0f / (tanHalfFov);
        matrix.m[2][2] = -(far + near) / (far - near);
        matrix.m[2][3] = -1.0f;
        matrix.m[3][2] = -(2.0f * far * near) / (far - near);
        return matrix;
    }

    static Matrix4 Translate(const Vector3& v) {
        Matrix4 matrix;
        matrix.m[3][0] = v.x;
        matrix.m[3][1] = v.y;
        matrix.m[3][2] = v.z;
        return matrix;
    }

    static Matrix4 Scale(const Vector3& v) {
        Matrix4 matrix;
        matrix.m[0][0] = v.x;
        matrix.m[1][1] = v.y;
        matrix.m[2][2] = v.z;
        return matrix;
    }

    static Matrix4 Rotate(float angleRad, Vector3 axis) {
        Matrix4 matrix;
        float c = cos(angleRad);
        float s = sin(angleRad);
        axis = axis.normalize();
        Vector3 t = axis * (1.0f - c);

        matrix.m[0][0] = c + t.x * axis.x;
        matrix.m[0][1] = t.x * axis.y + s * axis.z;
        matrix.m[0][2] = t.x * axis.z - s * axis.y;

        matrix.m[1][0] = t.y * axis.x - s * axis.z;
        matrix.m[1][1] = c + t.y * axis.y;
        matrix.m[1][2] = t.y * axis.z + s * axis.x;

        matrix.m[2][0] = t.z * axis.x + s * axis.y;
        matrix.m[2][1] = t.z * axis.y - s * axis.x;
        matrix.m[2][2] = c + t.z * axis.z;

        return matrix;
    }

    static Matrix4 LookAt(const Vector3& eye, const Vector3& target, const Vector3& EyeUp) {
        Vector3 CameraDirection = (target - eye).normalize();
        Vector3 CameraRight = CameraDirection.cross(EyeUp).normalize();
        Vector3 CameraUp = CameraRight.cross(CameraDirection);

        Matrix4 ViewMatrix(1.0f);
        ViewMatrix.m[0][0] = CameraRight.x;
        ViewMatrix.m[1][0] = CameraRight.y;
        ViewMatrix.m[2][0] = CameraRight.z;

        ViewMatrix.m[0][1] = CameraUp.x;
        ViewMatrix.m[1][1] = CameraUp.y;
        ViewMatrix.m[2][1] = CameraUp.z;

        ViewMatrix.m[0][2] = -CameraDirection.x;
        ViewMatrix.m[1][2] = -CameraDirection.y;
        ViewMatrix.m[2][2] = -CameraDirection.z;

        ViewMatrix.m[3][0] = -CameraRight.dot(eye);
        ViewMatrix.m[3][1] = -CameraUp.dot(eye);
        ViewMatrix.m[3][2] = CameraDirection.dot(eye);
        ViewMatrix.m[3][3] = 1.0f;

        return ViewMatrix;
    }

    static Matrix4 RotationMatrix(Math::Vector3& angle_vec) {
        float angle_x = Math::radians(angle_vec.x);
        float angle_y = Math::radians(angle_vec.y);
        float angle_z = Math::radians(angle_vec.z);

        double sin_x = sin(angle_x);
        double cos_x = cos(angle_x);
        double sin_y = sin(angle_y);
        double cos_y = cos(angle_y);
        double sin_z = sin(angle_z);
        double cos_z = cos(angle_z);

        return Math::Matrix4(
            cos_y * cos_z, (cos_z * sin_y * sin_x - sin_z * cos_x), (sin_z * sin_x + cos_x * cos_z * sin_y), 0.0f,
            sin_z * cos_y, (cos_z * cos_x + sin_y * sin_z * sin_x), (sin_y * sin_z * cos_x - sin_x * cos_z), 0.0f,
            -sin_y, cos_y * sin_x, cos_x * cos_y, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        );
    }

    static Matrix4 TranslationMatrix(Math::Vector3& position) {
        Matrix4 TMatrix(1.0f);
        TMatrix.m[0][3] = position.x;
        TMatrix.m[1][3] = position.y;
        TMatrix.m[2][3] = position.z;

        return TMatrix;
    }
};

}

#endif