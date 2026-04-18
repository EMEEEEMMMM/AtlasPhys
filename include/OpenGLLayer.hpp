#pragma once
#include <glad/glad.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "FrameBuffer.hpp"
#include "Shader.h"
#include "Camera.hpp"
#include "G_Objects.hpp"

enum class ProjectionMode { Perspective, Orthographic };

class OpenGLLayer {
public:
    OpenGLLayer(uint32_t width, uint32_t height);
    ~OpenGLLayer();

    void Render();
    void ResizeFrameBuffer(uint32_t width, uint32_t height);
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
    }
    
    static void SetUpGeometry(std::vector<float> vertices, std::vector<uint32_t> indices, GLuint& VAO, GLuint& VBO, GLuint& EBO);

    std::vector<G_Objects::P_Objects> objectslist;

private:
    Shader* m_Shader = nullptr;
    FrameBuffer* m_FrameBuffer = nullptr;

    ProjectionMode m_CurrentMode = ProjectionMode::Perspective;
    Camera m_Camera = Camera(Math::Vector3(0.0f, 0.0f, 3.0f));
};