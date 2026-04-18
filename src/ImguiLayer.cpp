#include <glad/glad.h>
#include "ImguiLayer.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

#include <Math/Vector3.hpp>
#include "G_Objects.hpp"

#include <iostream>

ImGuiLayer::ImGuiLayer(GLFWwindow* window, const char* glslVersion) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glslVersion);
}

ImGuiLayer::~ImGuiLayer() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::Begin() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
}

bool ImGuiLayer::RenderViewport(FrameBuffer* framebuffer) {
    bool resized = false;

    ImGui::Begin("ViewPort");
    {
        ImVec2 viewPortSize = ImGui::GetContentRegionAvail();
        
        if(viewPortSize.x * viewPortSize.y > 0 &&
            (static_cast<uint32_t>(viewPortSize.x) != framebuffer->GetWidth() ||
             static_cast<uint32_t>(viewPortSize.y) != framebuffer->GetHeight())
        )
        {
            resized = true;
            framebuffer->Resize(static_cast<uint32_t>(viewPortSize.x),
                                static_cast<uint32_t>(viewPortSize.y));
        }

        m_IsViewportFocused = ImGui::IsWindowFocused();
        m_IsViewportHovered = ImGui::IsWindowHovered();

        uint32_t textureID = framebuffer->GetColorAttachment();
        ImGui::Image(reinterpret_cast<void*>(static_cast<uintptr_t>(textureID)), 
                     viewPortSize, {0, 1}, {1, 0});
    }
    ImGui::End();

    if (show_demo_window)
    {
        ImGui::ShowDemoWindow(&show_demo_window); 
    }

    return resized;
}

void ImGuiLayer::RenderUI() {
    ImGui::Begin("Options");
    if (ImGui::Button("Add Object"))
    {
        ImGui::OpenPopup("AddObjects");
    }

    if (ImGui::BeginPopupModal("AddObjects", NULL, ImGuiWindowFlags_MenuBar))
    {
        static int item = 0;
        static float sideLength = 1.0f;
        static Math::Vector3 position = Math::Vector3();
        static float mass = 1.0f;
        static float restitution = 0.3f;
        static float color[4] = { 0.4f, 0.7f, 0.0f, 1.0f}; 

        ImGui::Combo("Shape", &item, "Cube\0Sphere\0\0");
        ImGui::InputFloat("Side Length", &sideLength, 0.1f, 2.0f, "%.2f");
        sideLength = ImClamp(sideLength, 0.01f, 100.0f);
        ImGui::InputFloat3("Position", &position.x);
        ImGui::InputFloat("Mass:", &mass, 0.1f, 0.5f, "%.1f");
        mass = ImClamp(mass, 0.0f, 1000.0f);
        ImGui::InputFloat("Restitution:", &restitution, 0.001f, 0.005f, "%.3f");
        restitution = ImClamp(restitution, 0.0f, 1.0f);
        ImGui::ColorEdit4("Color", color);

        if (ImGui::Button("Done"))
        {
            
            std::cout << item << position << " " << sideLength << " " << color[0] << " " << color[1] << " " << color[2] << " " << color[3] << " " << std::endl;
            if (item == 0) {
                std::cout << "Cube added" << std::endl;
                G_Objects::add_cube(sideLength, position, mass, restitution, color);
            } else if (item == 1){
                std::cout << "Sphere added" << std::endl;
                G_Objects::add_sphere(sideLength, position, mass, restitution, color);
            }
            
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Button("Close"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (ImGui::Button("Switch projection mode"))
    {
        if (SwitchProjection) {
            SwitchProjection();
        }
    }

    if (ImGui::Button("Add/Delete Coordinate Axis"))
    {
        // TODO
        std::cout << "Add / Delete Coordinate Axis" << std::endl;
    }
    
    if (ImGui::Button("Add / Delete the plane"))
    {
        // TODO
        std::cout << "Add / Delete the plane" << std::endl;
    }

    if (ImGui::Button("Start/stop the simulator"))
    {
        // TODO
        std::cout << "Start / stop" << std::endl;
    }

    if (ImGui::Button("Load / Unload the demo"))
    {
        // TODO
        std::cout << "Loaded / Unloaded the demo" << std::endl;
    }
    
    if (ImGui::Button("Shortcut to add a cube"))
    {
        // TODO
        std::cout << "Added a cube" << std::endl;
    }

    if (ImGui::Button("Shortcut to add a sphere"))
    {
        // TODO
        std::cout << "Added a sphere" << std::endl;
    }

    ImGui::Text("Properties");

    static float gravity = 9.800;
    ImGui::SliderFloat("Gravity", &gravity, 0.0f, 100.0f, "ratio = %.3f");

    static float mouseSensitivity = 1.0f;
    if (ImGui::SliderFloat("Mouse sensitivity", &mouseSensitivity, 1.0f, 2.0f, "ratio = %.4f")) {
        if (MouseSensitivityChange) {
            MouseSensitivityChange(mouseSensitivity);
        }
    }

    ImGui::End();
}

void ImGuiLayer::End(GLFWwindow* window) {
    ImGui::Render();

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow* backup = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup);
    }
}