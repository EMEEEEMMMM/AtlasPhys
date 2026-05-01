#include "OpenGLLayer.hpp"
#include "Math/Vector3.hpp"
#include "Math/Matrix4.hpp"
#include "Camera.hpp"
#include "G_Objects.hpp"
#include "Step.hpp"

#include <iostream>
#include <filesystem>
#include <GLFW/glfw3.h>
#include <vector>

OpenGLLayer::OpenGLLayer(uint32_t width, uint32_t height) {
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
	delete m_Shader;
	delete m_FrameBuffer;
}

void OpenGLLayer::SetUpGeometry(std::vector<float> vertices, std::vector<uint32_t> indices, GLuint& VAO, GLuint& VBO, GLuint& EBO) {
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3* sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
};

void OpenGLLayer::Render() {
	glEnable(GL_DEPTH_TEST);

	m_FrameBuffer->Bind();

	glViewport(0, 0, m_FrameBuffer->GetWidth(), m_FrameBuffer->GetHeight());
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m_Shader->Bind();

	float currentTime = glfwGetTime();
	deltaTime = currentTime - lastFrame;
	lastFrame = currentTime;

	// std::cout << deltaTime << std::endl;

	Math::Matrix4 view = m_Camera.GetViewMatrix();
	Math::Matrix4 projection = Math::Matrix4();
	float aspect = (float)m_FrameBuffer->GetWidth() / m_FrameBuffer->GetHeight();
	
	if (m_CurrentMode == ProjectionMode::Perspective) {
		projection = Math::Matrix4::Persp(Math::radians(m_Camera.Zoom), aspect, 0.1f, 100.0f);
	} else {
		float size = 5.0f;
		projection = Math::Matrix4::Ortho(-size * aspect, size * aspect, -size, size, 0.1f, 100.0f);
	}

	
	unsigned int viewLoc = glGetUniformLocation(m_Shader->m_RendererID, "view");
	unsigned int projectionLoc = glGetUniformLocation(m_Shader->m_RendererID, "projection");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view.m[0][0]);
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, &projection.m[0][0]);

	unsigned int modelLoc = glGetUniformLocation(m_Shader->m_RendererID, "model");

	Step::integrator(G_Objects::d_Objects, deltaTime);

	for (const auto& objPtr : G_Objects::g_Objects) {
		auto& obj = *objPtr;
		Math::Matrix4 model = obj.get_model_matrix();
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model.m[0][0]);
		glBindVertexArray(obj.VAO);
		glDrawElements(obj.GL_Type, obj.lenIndices, GL_UNSIGNED_INT, (void*)0);
		glBindVertexArray(0);
	}

	m_Shader->Unbind();
	m_FrameBuffer->UBind();
}

void OpenGLLayer::ResizeFrameBuffer(uint32_t width, uint32_t height) {
	m_FrameBuffer->Resize(width, height);
	glViewport(0, 0, width, height);
}

void OpenGLLayer::DeleteSingleObject(const std::shared_ptr<G_Objects::P_Objects>& obj) {
	glDeleteVertexArrays(1, &obj->VAO);
	glDeleteBuffers(1, &obj->VBO);
	glDeleteBuffers(1, &obj->EBO);
}

void OpenGLLayer::DrawCoordinateAxis() {
	std::vector<float> vertices = {
		// X-axis
		-100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
		100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
		
		// Y-axis
		0.0, -100.0, 0.0, 0.0, 1.0, 0.0, 1.0,
		0.0, 100.0, 0.0, 0.0, 1.0, 0.0, 1.0,
		
		// Z-axis
		0.0, 0.0, -100.0, 0.0, 1.0, 1.0, 1.0,
		0.0, 0.0, 100.0, 0.0, 1.0, 1.0, 1.0,
	};

	std::vector<uint32_t> indices = {
		0, 1,
		2, 3,
		4, 5
	};

	float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
	auto Axis = std::make_shared<G_Objects::P_Objects>(0.0, Math::Vector3(0.0f, 0.0f, 0.0f), 0.0, 0.0, color, GL_LINES, vertices, indices);
	SetUpGeometry(vertices, indices, Axis->VAO, Axis->VBO, Axis->EBO);
	G_Objects::g_Axis = *Axis;
	G_Objects::g_Objects.push_back(Axis);
}

void OpenGLLayer::DrawPlane() {
	std::vector<float> vertices = {
		-500, 0.0, 500, 0.5, 0.5, 0.5, 1.0,
		500, 0.0, 500, 0.5, 0.5, 0.5, 1.0,
		500, 0.0, -500, 0.5, 0.5, 0.5, 1.0,
		-500, 0.0, -500, 0.5, 0.5, 0.5, 1.0,

		-500, -2.0, 500, 0.5, 0.5, 0.5, 1.0,
		500, -2.0, 500, 0.5, 0.5, 0.5, 1.0,
		500, -2.0, -500, 0.5, 0.5, 0.5, 1.0,
		-500, -2.0, -500, 0.5, 0.5, 0.5, 1.0,
	};

	std::vector<uint32_t> indices = {
		0, 1, 2, 2, 3, 0,       
		4, 5, 6, 6, 7, 4,       
		0, 1, 5, 5, 4, 0,       
		2, 3, 7, 7, 6, 2,       
		1, 2, 6, 6, 5, 1,       
		3, 0, 4, 4, 7, 3
	};

	float color[4] = {0.5f, 0.5f, 0.5f, 1.0f};
	auto Plane = std::make_shared<G_Objects::P_Objects>(1000.0, Math::Vector3(0.0f, -1.0f, 0.0f), 0.0, 0.0, color, GL_TRIANGLES, vertices, indices);
	SetUpGeometry(vertices, indices, Plane->VAO, Plane->VBO, Plane->EBO);
	G_Objects::g_Plane = *Plane;
	G_Objects::g_Objects.push_back(Plane);	
}

void OpenGLLayer::RenderWithCustomView(FrameBuffer* fb, const Math::Matrix4& view, const Math::Matrix4& projection) {
	fb->Bind();
	glViewport(0, 0, fb->GetWidth(), fb->GetHeight());
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	m_Shader->Bind();

	// 传入自定义 view / projection
	glUniformMatrix4fv(glGetUniformLocation(m_Shader->m_RendererID, "view"),
					1, GL_FALSE, &view.m[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(m_Shader->m_RendererID, "projection"),
					1, GL_FALSE, &projection.m[0][0]);

	unsigned int modelLoc = glGetUniformLocation(m_Shader->m_RendererID, "model");

	// 与主 Render() 相同的物理积分（保持同步）
	Step::integrator(G_Objects::d_Objects, deltaTime);

	// 渲染全部对象（与主视口相同）
	for (const auto& objPtr : G_Objects::g_Objects) {
		Math::Matrix4 model = objPtr->get_model_matrix();
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, &model.m[0][0]);
		glBindVertexArray(objPtr->VAO);
		glDrawElements(objPtr->GL_Type, objPtr->lenIndices,
					GL_UNSIGNED_INT, nullptr);
	}

	m_Shader->Unbind();
	fb->UBind();
}