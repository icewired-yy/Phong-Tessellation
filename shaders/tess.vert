#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 vs_pos;
out vec3 vs_normal;

void main()
{
    vs_pos = aPos;
    vs_normal = aNormal;
}