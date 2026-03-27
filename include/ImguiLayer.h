#pragma once
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "FrameBuffer.h"

class ImGuiLayer {
public:
    ImGuiLayer(GLFWwindow* window, const char* glslVersion = "#version 130");
    ~ImGuiLayer();

    void Begin();
    bool RenderViewport(FrameBuffer* frameBuffer);
    void RenderUI();
    void End(GLFWwindow* window);
    ImGuiIO& GetIO() const { return ImGui::GetIO(); }
    bool show_demo_window = false;
};