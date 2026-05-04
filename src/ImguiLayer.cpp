#include "ImguiLayer.hpp"

#include <glad/glad.h>

#include <Math/Vector3.hpp>
#include <iostream>

#include "G_Objects.hpp"
#include "OpenGLLayer.hpp"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"

ImGuiLayer::ImGuiLayer(GLFWwindow* window, const char* glslVersion) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    io.IniFilename = nullptr;

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

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGuiWindowFlags rootFlags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    rootFlags |= ImGuiWindowFlags_NoBringToFrontOnFocus |
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove;
    rootFlags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::Begin("DockSpaceRoot", nullptr, rootFlags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    ImGuiID dockspaceId = ImGui::GetID("CustomDockSpace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

    static bool isFirstInit = true;

    if (isFirstInit) {
        isFirstInit = false;

        ImGui::DockBuilderRemoveNode(dockspaceId);
        ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

        ImGuiID mainNode = dockspaceId;
        ImGuiID leftNode, midNode, rightNode;
        ImGui::DockBuilderSplitNode(mainNode, ImGuiDir_Left, 0.2, &leftNode,
                                    &midNode);
        ImGui::DockBuilderSplitNode(midNode, ImGuiDir_Right, 0.25f, &rightNode,
                                    &midNode);

        ImGuiID topRight, bottomRight;
        ImGui::DockBuilderSplitNode(rightNode, ImGuiDir_Down, 0.5f,
                                    &bottomRight, &topRight);

        ImGui::DockBuilderDockWindow("Options", leftNode);
        ImGui::DockBuilderDockWindow("Viewport", midNode);
        ImGui::DockBuilderDockWindow("Dynamic Objects", topRight);
        ImGui::DockBuilderDockWindow("Object View", bottomRight);

        ImGui::DockBuilderFinish(dockspaceId);
    }

    ImGui::End();
}

bool ImGuiLayer::RenderViewport(FrameBuffer* framebuffer) {
    bool resized = false;

    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoCollapse);
    {
        ImVec2 viewPortSize = ImGui::GetContentRegionAvail();

        if (viewPortSize.x * viewPortSize.y > 0 &&
            (static_cast<uint32_t>(viewPortSize.x) != framebuffer->GetWidth() ||
             static_cast<uint32_t>(viewPortSize.y) !=
                 framebuffer->GetHeight())) {
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

    if (show_demo_window) {
        ImGui::ShowDemoWindow(&show_demo_window);
    }

    return resized;
}

void ImGuiLayer::RenderUI() {
    ImGui::Begin("Options", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse);
    if (ImGui::Button("Add Object")) {
        ImGui::OpenPopup("AddObjects");
    }

    if (ImGui::BeginPopupModal("AddObjects", NULL, ImGuiWindowFlags_MenuBar)) {
        static int item = 0;
        static float sideLength = 1.0f;
        static float width = 1.0f, height = 1.0f, depth = 1.0f;
        static float radius = 1.0f;
        static float baseRadius = 1.0f, pyramidHeight = 1.0f;
        static Math::Vector3 position = Math::Vector3();
        static float mass = 1.0f;
        static float restitution = 0.3f;
        static float color[4] = {1.0f, 0.0f, 0.0f, 1.0f};

        ImGui::Combo(
            "Shape", &item,
            "Cube\0Sphere\0Rectangular Prism\0Tetrahedron\0Square Pyramid\0"
            "Pentagonal Pyramid\0Hexagonal Pyramid\0Octahedron\0\0");

        ImGui::InputFloat3("Position", &position.x);
        ImGui::InputFloat("Mass:", &mass, 0.1f, 0.5f, "%.1f");
        mass = ImClamp(mass, 0.0f, 1000.0f);
        ImGui::InputFloat("Restitution:", &restitution, 0.001f, 0.005f, "%.3f");
        restitution = ImClamp(restitution, 0.0f, 1.0f);
        ImGui::ColorEdit4("Color", color);

        switch (item) {
            case 0:
                ImGui::InputFloat("Side Length", &sideLength, 0.1f, 2.0f,
                                  "%.2f");
                sideLength = ImClamp(sideLength, 0.01f, 100.0f);
                break;
            case 1:
                ImGui::InputFloat("Radius", &radius, 0.1f, 2.0f, "%.2f");
                radius = ImClamp(radius, 0.01f, 100.0f);
                break;
            case 2:
                ImGui::InputFloat("Width", &width, 0.1f, 2.0f, "%.2f");
                ImGui::InputFloat("Height", &height, 0.1f, 2.0f, "%.2f");
                ImGui::InputFloat("Depth", &depth, 0.1f, 2.0f, "%.2f");
                width = ImClamp(width, 0.01f, 100.0f);
                height = ImClamp(height, 0.01f, 100.0f);
                depth = ImClamp(depth, 0.01f, 100.0f);
                break;
            case 3:
                ImGui::InputFloat("Radius", &radius, 0.1f, 2.0f, "%.2f");
                radius = ImClamp(radius, 0.01f, 100.0f);
                break;
            case 4:
            case 5:
            case 6:
                ImGui::InputFloat("Base Radius", &baseRadius, 0.1f, 2.0f,
                                  "%.2f");
                ImGui::InputFloat("Height", &pyramidHeight, 0.1f, 2.0f, "%.2f");
                baseRadius = ImClamp(baseRadius, 0.01f, 100.0f);
                pyramidHeight = ImClamp(pyramidHeight, 0.01f, 100.0f);
                break;
            case 7:
                ImGui::InputFloat("Radius", &radius, 0.1f, 2.0f, "%.2f");
                radius = ImClamp(radius, 0.01f, 100.0f);
                break;
            default:
                break;
        }

        if (ImGui::Button("Done")) {
            switch (item) {
                case 0:
                    G_Objects::add_cube(sideLength, position, mass, restitution,
                                        color);
                    break;
                case 1:
                    G_Objects::add_sphere(radius, position, mass, restitution,
                                          color);
                    break;
                case 2:
                    G_Objects::add_rectangular_prism(width, height, depth,
                                                     position, mass,
                                                     restitution, color);
                    break;
                case 3:
                    G_Objects::add_tetrahedron(radius, position, mass,
                                               restitution, color);
                    break;
                case 4:
                    G_Objects::add_square_pyramid(baseRadius, pyramidHeight,
                                                  position, mass, restitution,
                                                  color);
                    break;
                case 5:
                    G_Objects::add_pentagonal_pyramid(baseRadius, pyramidHeight,
                                                      position, mass,
                                                      restitution, color);
                    break;
                case 6:
                    G_Objects::add_hexagonal_pyramid(baseRadius, pyramidHeight,
                                                     position, mass,
                                                     restitution, color);
                    break;
                case 7:
                    G_Objects::add_regular_octahedron(radius, position, mass,
                                                      restitution, color);
                    break;
                default:
                    break;
            }
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (ImGui::Button("Switch projection mode")) {
        if (SwitchProjection) {
            SwitchProjection();
        }
    }

    if (ImGui::Button("Add/Delete Coordinate Axis")) {
        if (AddOrDeleteCoordinateAxis) {
            AddOrDeleteCoordinateAxis();
        }
    }

    if (ImGui::Button("Add / Delete the plane")) {
        if (AddOrDeletePlane) {
            AddOrDeletePlane();
        }
    }

    if (ImGui::Button("Start/stop the simulator")) {
        // TODO
        std::cout << "Start / stop" << std::endl;
    }

    if (ImGui::Button("Load / Unload the demo")) {
        // TODO
        std::cout << "Loaded / Unloaded the demo" << std::endl;
    }

    if (ImGui::Button("Shortcut to add a cube")) {
        // TODO
        std::cout << "Added a cube" << std::endl;
    }

    if (ImGui::Button("Shortcut to add a sphere")) {
        // TODO
        std::cout << "Added a sphere" << std::endl;
    }

    if (ImGui::Button("Show d_Objects")) {
        show_dobject_window = !show_dobject_window;
    }

    ImGui::Text("Properties");

    static float gravity = -9.800;
    if (ImGui::SliderFloat("Gravity", &gravity, -100.0f, 0.0f,
                           "ratio = %.3f")) {
        if (GravityChange) {
            GravityChange(gravity);
        }
    };

    static float mouseSensitivity = 1.0f;
    if (ImGui::SliderFloat("Mouse sensitivity", &mouseSensitivity, 1.0f, 2.0f,
                           "ratio = %.4f")) {
        if (MouseSensitivityChange) {
            MouseSensitivityChange(mouseSensitivity);
        }
    }

    ImGui::End();

    if (show_dobject_window) {
        ImGuiWindowFlags dObjFlags = ImGuiWindowFlags_NoMove |
                                     ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoCollapse;
        ImGui::Begin("Dynamic Objects", &show_dobject_window, dObjFlags);

        int idx = 0;
        for (auto it = G_Objects::d_Objects.begin();
             it != G_Objects::d_Objects.end();) {
            if (auto obj = it->lock()) {
                ImGui::PushID(idx);
                bool selected = (selected_dobject == idx);
                ImGuiStyle& style = ImGui::GetStyle();
                float btnWidth = ImGui::CalcTextSize("Delete").x +
                                 style.FramePadding.x * 2.0f;

                float avail = ImGui::GetContentRegionAvail().x;
                float spacing = style.ItemSpacing.x;
                float selectableWidth = avail - btnWidth - spacing * 2.0f;
                if (selectableWidth < 0) selectableWidth = 0.0f;

                if (ImGui::Selectable(("Object " + std::to_string(idx)).c_str(),
                                      selected, 0,
                                      ImVec2(selectableWidth, 0.0f)))
                    selected_dobject = idx;

                ImGui::SameLine();
                ImGui::SetCursorPosX(avail - btnWidth);
                if (ImGui::Button("Delete",
                                  ImVec2(btnWidth, ImGui::GetFrameHeight()))) {
                    G_Objects::g_Objects.erase(
                        std::remove_if(
                            G_Objects::g_Objects.begin(),
                            G_Objects::g_Objects.end(),
                            [&](const std::shared_ptr<G_Objects::P_Objects>&
                                    p) { return p == obj; }),
                        G_Objects::g_Objects.end());

                    it = G_Objects::d_Objects.erase(it);
                    if (selected_dobject == idx) selected_dobject = -1;
                    ImGui::PopID();
                    continue;
                }

                if (selected) {
                    ImGui::Indent();
                    ImGui::Text("Pos: (%.2f, %.2f, %.2f)", obj->position.x,
                                obj->position.y, obj->position.z);
                    ImGui::Text("Velocity: (%.2f, %.2f, %.2f)", obj->velocity.x,
                                obj->velocity.y, obj->velocity.z);
                    ImGui::Text("Acceleration: (%.2f, %.2f, %.2f)",
                                obj->acceleration.x, obj->acceleration.y,
                                obj->acceleration.z);
                    ImGui::Text("Mass: %.2f", obj->mass);
                    ImGui::Unindent();
                }

                ImGui::PopID();
                ++it;
            } else {
                it = G_Objects::d_Objects.erase(it);
            }
            ++idx;
        }

        ImGui::End();

        ImGui::Begin("Object View", nullptr,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoCollapse);

        if (selected_dobject < 0) {
            ImGui::Text("Select an object");
        } else {
            std::shared_ptr<G_Objects::P_Objects> selObj;
            int idx = 0;
            for (auto& wk : G_Objects::d_Objects) {
                if (auto o = wk.lock()) {
                    if (idx == selected_dobject) {
                        selObj = o;
                        break;
                    }
                }
                ++idx;
            }

            if (selObj) {
                ImVec2 avail = ImGui::GetContentRegionAvail();
                if (avail.x > 0 && avail.y > 0) {
                    if (!m_ObjectViewFB) {
                        m_ObjectViewFB = new FrameBuffer((uint32_t)avail.x,
                                                         (uint32_t)avail.y);
                    } else if ((uint32_t)avail.x !=
                                   m_ObjectViewFB->GetWidth() ||
                               (uint32_t)avail.y !=
                                   m_ObjectViewFB->GetHeight()) {
                        m_ObjectViewFB->Resize((uint32_t)avail.x,
                                               (uint32_t)avail.y);
                    }

                    Math::Vector3 target = selObj->position;
                    float scaleFactor =
                        selObj->sideLength > 0.0f ? selObj->sideLength : 1.0f;
                    Math::Vector3 baseOffset(2.0f, 2.0f, 5.0f);
                    Math::Vector3 eye = target + baseOffset * scaleFactor;
                    Math::Matrix4 view = Math::Matrix4::LookAt(
                        eye, target, Math::Vector3(0.0f, 1.0f, 0.0f));

                    float aspect = (float)m_ObjectViewFB->GetWidth() /
                                   m_ObjectViewFB->GetHeight();
                    Math::Matrix4 proj;
                    if (m_OpenGLLayer->GetProjectionMode() ==
                        ProjectionMode::Perspective) {
                        proj = Math::Matrix4::Persp(
                            Math::radians(m_OpenGLLayer->GetCamera().Zoom),
                            aspect, 0.1f, 100.0f);
                    } else {
                        float size = 5.0f;
                        proj =
                            Math::Matrix4::Ortho(-size * aspect, size * aspect,
                                                 -size, size, 0.1f, 100.0f);
                    }

                    m_OpenGLLayer->RenderWithCustomView(m_ObjectViewFB, view,
                                                        proj);

                    uint32_t tex = m_ObjectViewFB->GetColorAttachment();
                    ImGui::Image(
                        reinterpret_cast<void*>(static_cast<uintptr_t>(tex)),
                        avail, ImVec2(0, 1), ImVec2(1, 0));
                }
            } else {
                ImGui::Text("Select an object");
            }
        }
        ImGui::End();
    }
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