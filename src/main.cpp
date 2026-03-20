#include <iostream>
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "ImguiLayer.h"
#include "OpenGLLayer.h"

void framebuffer_size_callback(GLFWwindow* window, int width ,int height) {
    glViewport(0, 0, width, height);
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "AtlasPhys", nullptr, nullptr);
    if (!window) 
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) 
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    OpenGLLayer renderLayer(800, 600);
    ImGuiLayer imguiLayer(window);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        renderLayer.Render();
        imguiLayer.Begin();

        if (imguiLayer.RenderViewport(renderLayer.GetFrameBuffer())) 
        {
            FrameBuffer* fb = renderLayer.GetFrameBuffer();
            glViewport(0, 0, fb->GetWidth(), fb->GetHeight());
        }

        imguiLayer.End(window);
        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}