#pragma once
#include <Math/Matrix4.hpp>
#include <Math/Vector3.hpp>
#include <Math/Utils.hpp>

#include <glad/glad.h>
#include <vector>

namespace G_Objects {
    struct P_Objects {
        float sideLength;
        float color[4];
        float mass;
        float restitution;

        bool collidable;

        int GL_Type;
        std::vector<float> vertices;
        std::vector<u_int32_t> indices;
        GLuint VAO, VBO, EBO;
        int lenIndices;

        Math::Vector3 position;
        Math::Vector3 velocity = Math::Vector3();
        Math::Vector3 acceleration = Math::Vector3();
        Math::Vector3 scale = Math::Vector3(1.0f, 1.0f, 1.0f);
        Math::Vector3 rotation = Math::Vector3();
        // RotationMatrix
        Math::Vector3 angularVelocity = Math::Vector3();

        P_Objects(float sideLength, Math::Vector3 position,
                float mass, float restitution, float color[4], int GL_Type, 
                std::vector<float> vertices, std::vector<u_int32_t> indices)
            : sideLength(sideLength),
            position(position),
            mass(mass),
            restitution(restitution),
            GL_Type(GL_Type),
            vertices(vertices),
            indices(indices) {
            this->collidable = true;
            this->lenIndices = indices.size();

            for (int i = 0; i < 4; i++) {
                this->color[i] = color[i];
            }

            this->VAO = 0;
            this->VBO = 0;
            this->EBO = 0;
        }

        Math::Matrix4 get_model_matrix() {
            Math::Matrix4 scaleMatrix = Math::Matrix4(scale.x, scale.y, scale.z, 1.0);
            Math::Matrix4 rotationMatrix = Math::Matrix4::RotationMatrix(rotation);
            Math::Matrix4 translationMatrix = Math::Matrix4::TranslationMatrix(position);

            return translationMatrix * rotationMatrix * scaleMatrix;
        }
    };

    extern std::vector<P_Objects> g_Objects;

    void add_cube(float sideLength, Math::Vector3 position,
                float mass, float restitution, float color[4]);
    void add_sphere(float sideLength, Math::Vector3 position,
                float mass, float restitution, float color[4]);
}