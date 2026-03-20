#include "OpenGLLayer.h"

#include <iostream>
#include <filesystem>

OpenGLLayer::OpenGLLayer(uint32_t width, uint32_t height) {
    InitGeometry();
	std::filesystem::path baseDir = 
		std::filesystem::path(__FILE__)
			.parent_path()
			.parent_path();

	std::string vertPath = (baseDir / "src/ShaderProgram/vertex_shader.vert").string();
    std::string fragPath = (baseDir / "src/ShaderProgram/fragment_shader.frag").string();
    
	m_Shader = new Shader(vertPath, fragPath);
	m_FrameBuffer = new FrameBuffer(width, height);
}

OpenGLLayer::~OpenGLLayer() {
	glDeleteVertexArrays(1, &m_VAO);
	glDeleteBuffers(1, &m_VBO);
	delete m_Shader;
	delete m_FrameBuffer;
}

void OpenGLLayer::InitGeometry() {
	float positions[6] = {
		-0.5f, -0.5,
		0.0f, 0.5f,
		0.5f, -0.5f
	};

	glGenVertexArrays(1, &m_VAO);
	glBindVertexArray(m_VAO);

	glGenBuffers(1, &m_VBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);
};

void OpenGLLayer::Render() {
	m_FrameBuffer->Bind();

	glViewport(0, 0, m_FrameBuffer->GetWidth(), m_FrameBuffer->GetHeight());
	glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m_Shader->Bind();
	glBindVertexArray(m_VAO);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindVertexArray(0);
	m_Shader->Unbind();

	m_FrameBuffer->UBind();
}

void OpenGLLayer::ResizeFrameBuffer(uint32_t width, uint32_t height) {
	m_FrameBuffer->Resize(width, height);
	glViewport(0, 0, width, height);
}