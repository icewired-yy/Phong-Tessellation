#version 410 core

// Input vertex data (from VBO)
layout (location = 0) in vec3 aPos;

// Uniform variable for the Model-View-Projection matrix
uniform mat4 mvp;

void main()
{
    // Transform the vertex position and assign it to the output
    gl_Position = mvp * vec4(aPos, 1.0);
}