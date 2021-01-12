#version 330 core
in float aoNumber1;
out vec4 FragColor;
void main()
{
   float r = max(min(aoNumber1, 1.0f), 0.0f);
   float g = max(min(aoNumber1, 1.0f), 0.0f);
   float b = max(min(aoNumber1, 1.0f), 0.0f);
   FragColor = vec4(r,g,b, 1.0f);
}