#pragma once
#include <GL/glew.h>

#include <cstdint>

#include "FrameBuffer.h"
#include "Shader.h"

class OpenGLLayer {
   public:
    OpenGLLayer(uint32_t width, uint32_t height);
    ~OpenGLLayer();

    void Render();
    void ResizeFrameBuffer(uint32_t width, uint32_t height);
    FrameBuffer* GetFrameBuffer() const { return m_FrameBuffer; };

   private:
    void InitGeometry();
    GLuint m_VAO = 0;
    GLuint m_VBO = 0;
    Shader* m_Shader = nullptr;
    FrameBuffer* m_FrameBuffer = nullptr;
};