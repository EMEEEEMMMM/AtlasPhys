#version 330 compatibility
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;

out vec4 vertexColor;

uniform mat4 mvp;

void main(){
    gl_Position = ftransform();
    vertexColor = aColor;
}

