#version 330 core
in vec4 vertexColor;

out vec4 FragColor;

uniform vec4 lightColor;

void main(){
    FragColor = vertexColor;   
}