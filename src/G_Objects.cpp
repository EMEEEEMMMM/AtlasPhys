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
                0 - (sideLength / 2), 0 + (sideLength / 2), 0 + (sideLength / 2), color[0], color[1], color[2], color[3],
                0 + (sideLength / 2), 0 + (sideLength / 2), 0 + (sideLength / 2), color[0], color[1], color[2], color[3],
                0 + (sideLength / 2), 0 - (sideLength / 2), 0 + (sideLength / 2), color[0], color[1], color[2], color[3],
                0 - (sideLength / 2), 0 - (sideLength / 2), 0 + (sideLength / 2), color[0], color[1], color[2], color[3],
                0 - (sideLength / 2), 0 + (sideLength / 2), 0 - (sideLength / 2), color[0], color[1], color[2], color[3],
                0 + (sideLength / 2), 0 + (sideLength / 2), 0 - (sideLength / 2), color[0], color[1], color[2], color[3],
                0 + (sideLength / 2), 0 - (sideLength / 2), 0 - (sideLength / 2), color[0], color[1], color[2], color[3],
                0 - (sideLength / 2), 0 - (sideLength / 2), 0 - (sideLength / 2), color[0], color[1], color[2], color[3],
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
            cube->localNormals = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f}
            };

            cube->localEdges = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f}
            };
            
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

                    float realX = relX + 0;
                    float realY = relY + 0;
                    float realZ = relZ + 0;

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

    void add_rectangular_prism(
        float width, float height, float depth, Math::Vector3 position, 
        float mass, float restitution, float color[4]) {
            float halfW = width / 2.0f;
            float halfH = height / 2.0f;
            float halfD = depth / 2.0f;

            std::vector<float> vertices = {
                0 - halfW, 0 + halfH, 0 + halfD, color[0], color[1], color[2], color[3],
                0 + halfW, 0 + halfH, 0 + halfD, color[0], color[1], color[2], color[3],
                0 + halfW, 0 - halfH, 0 + halfD, color[0], color[1], color[2], color[3],
                0 - halfW, 0 - halfH, 0 + halfD, color[0], color[1], color[2], color[3],
                0 - halfW, 0 + halfH, 0 - halfD, color[0], color[1], color[2], color[3],
                0 + halfW, 0 + halfH, 0 - halfD, color[0], color[1], color[2], color[3],
                0 + halfW, 0 - halfH, 0 - halfD, color[0], color[1], color[2], color[3],
                0 - halfW, 0 - halfH, 0 - halfD, color[0], color[1], color[2], color[3],
            };

            std::vector<u_int32_t> indices = {
                0, 1, 2, 0, 2, 3,
                4, 5, 1, 4, 1, 0,
                3, 2, 6, 3, 6, 7,
                5, 4, 7, 5, 7, 6,
                1, 5, 6, 1, 6, 2,
                4, 0, 3, 4, 3, 7
            };

            float characteristic_size = std::max({width, height, depth});
            auto prism = std::make_shared<P_Objects>(characteristic_size, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
            
            prism->localNormals = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f}
            };

            prism->localEdges = {
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f},
                {0.0f, 0.0f, 1.0f}
            };

            OpenGLLayer::SetUpGeometry(vertices, indices, prism->VAO, prism->VBO, prism->EBO);
            g_Objects.push_back(prism);
            d_Objects.push_back(prism); 
        }

    void add_tetrahedron(
        float radius, Math::Vector3 position, 
        float mass, float restitution, float color[4]) {
            Math::Vector3 baseVerts[4] = {
                {1.0f, 1.0f, 1.0f}, {-1.0f, -1.0f, 1.0f},
                {-1.0f, 1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}
            };

            std::vector<float> vertices;
            for (int i = 0; i < 4; ++i) {
                Math::Vector3 v = baseVerts[i].normalize() * radius;

                vertices.insert(vertices.end(), {v.x, v.y, v.z, color[0], color[1], color[2], color[3]});
            }

            std::vector<u_int32_t> indices = {0,1,2, 0,2,3, 0,3,1, 1,3,2};

            auto tetra = std::make_shared<P_Objects>(radius, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
            
            Math::Vector3 v0(vertices[0], vertices[1], vertices[2]), 
                          v1(vertices[7], vertices[8], vertices[9]), 
                          v2(vertices[14], vertices[15], vertices[16]),
                          v3(vertices[21], vertices[22], vertices[23]);

            tetra->localNormals = {
                (v1 - v0).cross(v2 - v0).normalize(),
                (v2 - v0).cross(v3 - v0).normalize(),
                (v3 - v0).cross(v1 - v0).normalize(),
                (v2 - v1).cross(v3 - v1).normalize()
            };

            tetra->localEdges = {
                v1 - v0, v2 - v0, v3 - v0, v2 - v1 , v3 - v2, v1 - v3
            };

            OpenGLLayer::SetUpGeometry(vertices, indices, tetra->VAO, tetra->VBO, tetra->EBO);
            g_Objects.push_back(tetra);
            d_Objects.push_back(tetra);
        }

    void add_square_pyramid(
        float base_radius, float height, Math::Vector3 position,
        float mass, float restitution, float color[4]) {
            float yBottom = 0 - height/2.0f;
            float yTop = 0 + height / 2.0f;

            std::vector<float> vertices;

            Math::Vector3 baseVerts[4] = {
                {base_radius, yBottom, base_radius}, {-base_radius, yBottom, base_radius},
                {-base_radius, yBottom, -base_radius}, {base_radius, yBottom, -base_radius}
            };
            for (auto& bv : baseVerts) {
                Math::Vector3 v(bv.x + 0, bv.y, bv.z + 0);
                vertices.insert(vertices.end(), {v.x, v.y, v.z, color[0], color[1], color[2], color[3]});
            }

            vertices.insert(vertices.end(), {0, yTop, 0, color[0], color[1], color[2], color[3]});

            std::vector<u_int32_t> indices = {
                0,1,2, 0,2,3,
                0,1,4, 1,2,4, 2,3,4, 3,0,4
            };

            float charSize = std::max(base_radius, height);
            auto pyramid = std::make_shared<P_Objects>(charSize, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
            
            Math::InitializePyramidAxes(pyramid, 4);

            OpenGLLayer::SetUpGeometry(vertices, indices, pyramid->VAO, pyramid->VBO, pyramid->EBO);
            g_Objects.push_back(pyramid);
            d_Objects.push_back(pyramid);
        }

    void add_pentagonal_pyramid(
        float base_radius, float height, Math::Vector3 position,
        float mass, float restitution, float color[4]) {
            float yBottom = 0 - height/2.0f;
            float yTop = 0 + height/2.0f;

            std::vector<float> vertices;
            const int base_sides = 5;
            for (int i = 0; i < base_sides; ++i) {
                float angle = 2.0f * M_PI * i / base_sides;
                float x = base_radius * cos(angle) + 0;
                float z = base_radius * sin(angle) + 0;
                vertices.insert(vertices.end(), {x, yBottom, z, color[0], color[1], color[2], color[3]});
            }
            vertices.insert(vertices.end(), {0, yTop, 0, color[0], color[1], color[2], color[3]});

            std::vector<u_int32_t> indices = {
                0,1,2, 0,2,3, 0,3,4,
                0,1,5, 1,2,5, 2,3,5, 3,4,5, 4,0,5
            };

            float char_size = std::max(base_radius, height);
            auto pyramid = std::make_shared<P_Objects>(char_size, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
            
            Math::InitializePyramidAxes(pyramid, 5);
            
            OpenGLLayer::SetUpGeometry(vertices, indices, pyramid->VAO, pyramid->VBO, pyramid->EBO);
            g_Objects.push_back(pyramid);
            d_Objects.push_back(pyramid);   
        }
    
    void add_hexagonal_pyramid(
        float base_radius, float height, Math::Vector3 position,
        float mass, float restitution, float color[4]) {
            float yBottom = 0 - height/2.0f;
            float yTop = 0 + height/2.0f;

            std::vector<float> vertices;
            const int base_sides = 6;
            for (int i = 0; i < base_sides; ++i) {
                float angle = 2.0f * M_PI * i / base_sides;
                float x = base_radius * cos(angle) + 0;
                float z = base_radius * sin(angle) + 0;
                vertices.insert(vertices.end(), {x, yBottom, z, color[0], color[1], color[2], color[3]});
            }
            vertices.insert(vertices.end(), {0, yTop, 0, color[0], color[1], color[2], color[3]});

            std::vector<u_int32_t> indices = {
                0,1,2, 0,2,3, 0,3,4, 0,4,5,
                0,1,6, 1,2,6, 2,3,6, 3,4,6, 4,5,6, 5,0,6
            };

            float char_size = std::max(base_radius, height);
            auto pyramid = std::make_shared<P_Objects>(char_size, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
            
            Math::InitializePyramidAxes(pyramid, 6);
            
            OpenGLLayer::SetUpGeometry(vertices, indices, pyramid->VAO, pyramid->VBO, pyramid->EBO);
            g_Objects.push_back(pyramid);
            d_Objects.push_back(pyramid);   
        }

    void add_regular_octahedron(
        float radius, Math::Vector3 position,
        float mass, float restitution, float color[4]) {
            Math::Vector3 base_verts[6] = {
                {radius, 0, 0}, {-radius, 0, 0},
                {0, radius, 0}, {0, -radius, 0},
                {0, 0, radius}, {0, 0, -radius}
            };

            std::vector<float> vertices;
            for (auto& bv : base_verts) {
                Math::Vector3 v = bv;
                vertices.insert(vertices.end(), {v.x, v.y, v.z, color[0], color[1], color[2], color[3]});
            }

            std::vector<u_int32_t> indices = {
                0,2,4, 0,4,3, 0,3,5, 0,5,2,
                1,4,2, 1,3,4, 1,5,3, 1,2,5
            };

            auto octa = std::make_shared<P_Objects>(radius, position, mass, restitution, color, GL_TRIANGLES, vertices, indices);
            
            Math::Vector3 v0 = Math::GetVertsFromVertices(vertices, 0);
            Math::Vector3 v2 = Math::GetVertsFromVertices(vertices, 2);
            Math::Vector3 v4 = Math::GetVertsFromVertices(vertices, 4);
        
            octa->localEdges.push_back(v2 - v0);
            octa->localEdges.push_back(v4 - v2);
            octa->localEdges.push_back(v0 - v4);
            
            int faceIndices[4][3] = {{0,2,4}, {0,4,3}, {0,3,5}, {0,5,2}};
            for (int i = 0; i < 4; ++i) {
                Math::Vector3 a = Math::GetVertsFromVertices(vertices, faceIndices[i][0]);
                Math::Vector3 b = Math::GetVertsFromVertices(vertices, faceIndices[i][1]);
                Math::Vector3 c = Math::GetVertsFromVertices(vertices, faceIndices[i][2]);
                octa->localNormals.push_back((b - a).cross(c - a).normalize());
            }
            
            OpenGLLayer::SetUpGeometry(vertices, indices, octa->VAO, octa->VBO, octa->EBO);
            g_Objects.push_back(octa);
            d_Objects.push_back(octa);
        }
}