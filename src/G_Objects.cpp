#include "G_Objects.hpp"
#include <Math/Utils.hpp>
#include "OpenGLLayer.hpp"

#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

namespace G_Objects {
    
    std::vector<std::shared_ptr<P_Objects>> g_Objects;
    std::vector<std::weak_ptr<P_Objects>> d_Objects;

    P_Objects g_Axis;
    P_Objects g_Plane;

    void add_cube(
        float sideLength, Math::Vector3 position,
        float mass, float restitution, float color[4]) {
        std::vector<float> vertices = {
            position.x - (sideLength / 2), position.y + (sideLength / 2), position.z + (sideLength / 2), color[0], color[1], color[2], color[3],
            position.x + (sideLength / 2), position.y + (sideLength / 2), position.z + (sideLength / 2), color[0], color[1], color[2], color[3],
            position.x + (sideLength / 2), position.y - (sideLength / 2), position.z + (sideLength / 2), color[0], color[1], color[2], color[3],
            position.x - (sideLength / 2), position.y - (sideLength / 2), position.z + (sideLength / 2), color[0], color[1], color[2], color[3],
            position.x - (sideLength / 2), position.y + (sideLength / 2), position.z - (sideLength / 2), color[0], color[1], color[2], color[3],
            position.x + (sideLength / 2), position.y + (sideLength / 2), position.z - (sideLength / 2), color[0], color[1], color[2], color[3],
            position.x + (sideLength / 2), position.y - (sideLength / 2), position.z - (sideLength / 2), color[0], color[1], color[2], color[3],
            position.x - (sideLength / 2), position.y - (sideLength / 2), position.z - (sideLength / 2), color[0], color[1], color[2], color[3],
        };
        std::vector<u_int32_t> indices = {
            0, 1, 2, 0, 2, 3,
            4, 5, 1, 4, 1, 0,
            3, 2, 6, 3, 6, 7,
            5, 4, 7, 5, 7, 6,
            1, 5, 6, 1, 6, 2,
            4, 0, 3, 4, 3, 7
        };

        auto cube = std::make_shared<P_Objects>(sideLength, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
        OpenGLLayer::SetUpGeometry(vertices, indices, cube->VAO, cube->VBO, cube->EBO);
        g_Objects.push_back(cube);
        d_Objects.push_back(cube);   
    }

    void add_sphere(
        float sideLength, Math::Vector3 position,
        float mass, float restitution, float color[4]) {
        
        int rings = 100;
        int sectors = 100;
        std::vector<float> vertices;
        std::vector<u_int32_t> indices;

        std::vector<float> phi(rings + 1);
        const float phiStart = -static_cast<float>(M_PI) / 2.0f;
        const float phiEnd = static_cast<float>(M_PI) / 2.0f;
        
        for (int i = 0; i <= rings; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(rings);
            phi[i] = phiStart * (1.0f - t) + phiEnd * t;
        }

        std::vector<float> theta(sectors + 1);
        const float thetaStart = 0.0f;
        const float thetaEnd = 2.0f * static_cast<float>(M_PI);
        for (int i = 0; i <= sectors; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(sectors);
            theta[i] = thetaStart * (1.0f - t) + thetaEnd * t;
        }

        vertices.reserve((rings + 1) * (sectors + 1) * 7);
        for (int i = 0; i <= rings; ++i) {
            float cosPhi = std::cos(phi[i]);
            float sinPhi = std::sin(phi[i]);

            for (int j = 0; j <= sectors; ++j) {
                float cosTheta = std::cos(theta[j]);
                float sinTheta = std::sin(theta[j]);

                float relX = sideLength * cosPhi * cosTheta;
                float relY = sideLength * sinPhi;
                float relZ = sideLength * cosPhi * sinTheta;

                float realX = relX + position.x;
                float realY = relY + position.y;
                float realZ = relZ + position.z;

                vertices.push_back(realX);
                vertices.push_back(realY);
                vertices.push_back(realZ);
                vertices.push_back(color[0]);
                vertices.push_back(color[1]);
                vertices.push_back(color[2]);
                vertices.push_back(color[3]);
            }
        }

        indices.reserve(rings * sectors * 6);
        for (int ring = 0; ring < rings; ++ring) {
            for (int sector = 0; sector < sectors; ++sector) {
                int idx = ring * (sectors + 1) + sector;
                int idx1 = (ring + 1) * (sectors + 1) + sector;
                int idx2 = (ring + 1) * (sectors + 1) + (sector + 1);
                int idx3 = ring * (sectors + 1) + (sector + 1);

                indices.push_back(idx);
                indices.push_back(idx1);
                indices.push_back(idx2);

                indices.push_back(idx);
                indices.push_back(idx2);
                indices.push_back(idx3);
            }
        }

        auto sphere = std::make_shared<P_Objects>(sideLength, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
        OpenGLLayer::SetUpGeometry(vertices, indices, sphere->VAO, sphere->VBO, sphere->EBO);
        g_Objects.push_back(sphere);
        d_Objects.push_back(sphere);
    }
}