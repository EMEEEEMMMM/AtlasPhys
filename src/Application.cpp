#include "Application.hpp"
#include <iostream>
#include <GLFW/glfw3.h>


Application::Application(int width, int height, const char* title)
    : m_Width(width), m_Height(height), m_Title(title) {}

Application::~Application() {
    CleanUp();
}

bool Application::Init() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // For mac os:
    // #ifdef __APPLE__
    //     glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    // #endif

    m_Window = glfwCreateWindow(m_Width, m_Height, m_Title.c_str(), nullptr, nullptr);
    if (!m_Window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_Window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return false;
    }

    glViewport(0, 0, m_Width, m_Height);

    m_OpenGLLayer = std::make_unique<OpenGLLayer>(m_Width, m_Height);
    m_ImGuiLayer = std::make_unique<ImGuiLayer>(m_Window);

    m_ImGuiLayer->SwitchProjection = [this]() {
        m_OpenGLLayer->ToggleProjectionMode();
    };

    return true;
}

void Application::MainLoop() {
    while (!glfwWindowShouldClose(m_Window)) {
        glfwPollEvents();

        m_OpenGLLayer->Render();

        m_ImGuiLayer->Begin();

        if (m_ImGuiLayer->RenderViewport(m_OpenGLLayer->GetFrameBuffer())) {
            auto fb = m_OpenGLLayer->GetFrameBuffer();
            glViewport(0, 0, fb->GetWidth(), fb->GetHeight());
        }

        m_ImGuiLayer->RenderUI();
        m_ImGuiLayer->End(m_Window);

        glfwSwapBuffers(m_Window);
    }
}

void Application::CleanUp() {
    m_ImGuiLayer.reset();
    m_OpenGLLayer.reset();

    if (m_Window) {
        glfwDestroyWindow(m_Window);
        m_Window = nullptr;
    }
    glfwTerminate();
}

int Application::Run() {
    if (!Init()) return -1;
    MainLoop();
    return 0;
}