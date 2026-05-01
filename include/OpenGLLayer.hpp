#pragma once
#include <glad/glad.h>

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

#include "FrameBuffer.hpp"
#include "Shader.h"
#include "Camera.hpp"
#include "G_Objects.hpp"

enum class ProjectionMode { Perspective, Orthographic };
static bool m_CoordinateAxis = false;
static bool m_Plane = false;

class OpenGLLayer {
public:
    OpenGLLayer(uint32_t width, uint32_t height);
    ~OpenGLLayer();

    void Render();
    void ResizeFrameBuffer(uint32_t width, uint32_t height);
    void DeleteSingleObject(const std::shared_ptr<G_Objects::P_Objects>&);
    void DrawCoordinateAxis();
    void DrawPlane();
    FrameBuffer* GetFrameBuffer() const { return m_FrameBuffer; };

    void ToggleProjectionMode() {
        if (m_CurrentMode == ProjectionMode::Perspective) {
            m_CurrentMode = ProjectionMode::Orthographic;
            std::cout << "Switch to Orthographic projection" << std::endl;
        } else {
            m_CurrentMode = ProjectionMode::Perspective;
            std::cout << "Switch to Perspective projection" << std::endl;
        }
    }

    Camera& GetCamera() { return m_Camera; } 

    void SensitivityChange(float mouseSensitivity) {
        m_Camera.MouseSensitivity = mouseSensitivity;
        std::cout<< "Switch to: " << m_Camera.MouseSensitivity << " sensitivity." << std::endl;
    }

    void GravityChange(float gravity) {
        Math::GRAVITY = gravity;
        std::cout<< "Switch to: " << Math::GRAVITY << " gravity." << std::endl;
    }

    void CoordinateAxisDraw() {
        if (m_CoordinateAxis) {
            auto it = std::find_if(G_Objects::g_Objects.begin(), 
                                   G_Objects::g_Objects.end(), 
                                    [](const std::shared_ptr<G_Objects::P_Objects>& obj) { 
                                        return obj->VAO == G_Objects::g_Axis.VAO;
                                    });
            if (it != G_Objects::g_Objects.end()) {
                DeleteSingleObject(*it);
                G_Objects::g_Objects.erase(it);
            }

            G_Objects::g_Axis = G_Objects::P_Objects();
            m_CoordinateAxis = false;
        } else {
            m_CoordinateAxis = true;
            DrawCoordinateAxis();
        }
    }

    void PlaneDraw() {
        if (m_Plane) {
            auto it = std::find_if(G_Objects::g_Objects.begin(), 
                                   G_Objects::g_Objects.end(), 
                                    [](const std::shared_ptr<G_Objects::P_Objects>& obj) { 
                                        return obj->VAO == G_Objects::g_Plane.VAO;
                                    });
            if (it != G_Objects::g_Objects.end()) {
                DeleteSingleObject(*it);
                G_Objects::g_Objects.erase(it);
            }

            G_Objects::g_Plane = G_Objects::P_Objects();
            m_Plane = false;
        } else {
            m_Plane = true;
            DrawPlane();
        }
    }
    
    static void SetUpGeometry(std::vector<float> vertices, std::vector<uint32_t> indices, GLuint& VAO, GLuint& VBO, GLuint& EBO);

    ProjectionMode GetProjectionMode() const { return m_CurrentMode; }
    float GetCameraZoom() const { return m_Camera.Zoom; }

    void RenderWithCustomView(FrameBuffer* fb, const Math::Matrix4& view, const Math::Matrix4& projection);


private:
    Shader* m_Shader = nullptr;
    FrameBuffer* m_FrameBuffer = nullptr;

    float deltaTime = 0.0f;
    float lastFrame = 0.0f;

    ProjectionMode m_CurrentMode = ProjectionMode::Perspective;
    Camera m_Camera = Camera(Math::Vector3(0.0f, 0.0f, 3.0f));
};