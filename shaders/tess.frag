#version 410 core

in vec3 te_pos_world;
in vec3 te_normal_world;

out vec4 FragColor;

uniform vec3 viewPos; // Camera position in world space

void main()
{
    vec3 lightPos = vec3(10, 10, 10);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    vec3 objectColor = vec3(0.8, 0.8, 0.8);

    // Ambient
    float ambientStrength = 0.2;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(te_normal_world);
    vec3 lightDir = normalize(lightPos - te_pos_world);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - te_pos_world);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}