#pragma once
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <functional>

#include "FrameBuffer.hpp"
#include "imgui.h"

class OpenGLLayer;

class ImGuiLayer {
   public:
    ImGuiLayer(GLFWwindow* window, const char* glslVersion = "#version 130");
    ~ImGuiLayer();

    void Begin();
    bool RenderViewport(FrameBuffer* frameBuffer);
    void RenderUI();
    void End(GLFWwindow* window);

    std::function<void()> SwitchProjection;
    std::function<void(float)> MouseSensitivityChange;
    std::function<void()> AddOrDeleteCoordinateAxis;
    std::function<void()> AddOrDeletePlane;
    std::function<void(float)> GravityChange;

    ImGuiIO& GetIO() const { return ImGui::GetIO(); }
    bool show_demo_window = false;

    bool IsViewportFocused() const { return m_IsViewportFocused; }
    bool IsViewportHovered() const { return m_IsViewportHovered; }

    void SetOpenGLLayer(OpenGLLayer* layer) { m_OpenGLLayer = layer; }

   private:
    bool projection_mode = true;

    bool m_IsViewportFocused = false;
    bool m_IsViewportHovered = false;

    bool show_dobject_window = true;
    int selected_dobject = -1;

    FrameBuffer* m_ObjectViewFB = nullptr;
    OpenGLLayer* m_OpenGLLayer = nullptr;
};