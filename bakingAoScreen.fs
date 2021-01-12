#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

const float offset = 1.0 / 1024.0;  

void main()
{
    vec2 offsets[25] = vec2[](
		vec2(-2*offset,  2*offset), // 左上
		vec2(-offset,  2*offset), // 左上
        vec2( 0.0f,    2*offset), // 正上
        vec2( offset,  2*offset), // 右上
		vec2( 2*offset,  2*offset), // 右上
        vec2(-2*offset,  offset), // 左上
		vec2(-offset,  offset), // 左上
        vec2( 0.0f,    offset), // 正上
        vec2( offset,  offset), // 右上
		vec2( 2*offset,  offset), // 右上

		vec2(-2*offset,  0.0f),   // 左
        vec2(-offset,  0.0f),   // 左
        vec2( 0.0f,    0.0f),   // 中
        vec2( offset,  0.0f),   // 右
		vec2( 2*offset,  0.0f),   // 右

        vec2(-2*offset, -offset), // 左下
		vec2(-offset, -offset), // 左下
        vec2( 0.0f,   -offset), // 正下
        vec2( offset, -offset),  // 右下
		vec2( 2*offset, -offset),  // 右下
		vec2(-2*offset, -2*offset), // 左下
		vec2(-offset, -2*offset), // 左下
        vec2( 0.0f,   -2*offset), // 正下
        vec2( offset, -2*offset),  // 右下
		vec2( 2*offset, -2*offset)  // 右下
    );
	int num=44;
    float kernel[25] = float[](
    0.0 / num, 1.0 / num, 2.0 / num,1.0 / num,0.0 / num,
    1.0 / num, 2.0 / num, 4.0 / num,2.0 / num,1.0 / num,
    2.0 / num, 4.0 / num, 8.0 / num,4.0 / num,2.0 / num,
	1.0 / num, 2.0 / num, 4.0 / num,2.0 / num,1.0 / num,
	0.0 / num, 1.0 / num, 2.0 / num,1.0 / num,0.0 / num  
);


    vec3 sampleTex[25];
	int num1=0;
	float f=0.0;
    for(int i = 0; i < 25; i++)
    {
        sampleTex[i] = vec3(texture(screenTexture, TexCoords.st + offsets[i]));
		if(sampleTex[i].z<0.9f){
			num1++;
			f=max(sampleTex[i].z,f);
		}
    }
    vec3 col = vec3(0.0);
	if(num1>0&&sampleTex[12].x==1.0f){
		FragColor = vec4(vec3(f), 1.0);
	}else{
		FragColor = texture(screenTexture, TexCoords);
	}
}