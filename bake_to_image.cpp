

#include <fstream>
#include <map>
#include "bake_to_image.h"
#include "include/glad/glad.h"
#include "include/GLFW/glfw3.h"
//如果使用osmesa的话 还需要调用glfw3native
#if USE_NATIVE_OSMESA
#define GLFW_EXPOSE_NATIVE_OSMESA
#include "../include/GLFW/glfw3native.h"
#endif
#include "toojpeg.h"
#include "Shader.h"

std::string OUT_FILE="ao.jpg";
int WINDOWS_SIZE =2048;
namespace {
	const char *bakingAoShaderVS = "#version 330 core\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"layout (location = 0) in vec3 aPos;\n"
			"layout (location = 1) in float aoNumber;\n"
			"out float aoNumber1;\n"
			"void main()\n"
			"{\n"
			"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
			"aoNumber1=aoNumber;\n"
			"}\0";
	const char *bakingAoShaderFS = "#version 330 core\n"
			"out vec4 FragColor;\n"
			"in float aoNumber1;\n"
			"void main()\n"
			"{\n"
			"	float r = max(min(aoNumber1, 1.0f), 0.0f);\n"
			"	float g = max(min(aoNumber1, 1.0f), 0.0f);\n"
			"	float b = max(min(aoNumber1, 1.0f), 0.0f);\n"
			"   FragColor = vec4(r, g, b, 1.0f);\n"
			"}\n\0";
	const char *bakingAoScreenVS = "#version 330 core\n"
			"layout (location = 0) in vec3 aPos;\n"
			"layout (location = 1) in vec2 aTexCoords;\n"
			"out vec2 TexCoords;\n"
			"void main()\n"
			"{\n"
			"	TexCoords = aTexCoords;\n"
			"   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0); \n"
			"}\0";
	const char *bakingAoScreenFS = "#version 330 core\n"
			"out vec4 FragColor;\n"
			"in vec2 TexCoords;\n"
			"uniform sampler2D screenTexture;\n"
			"const float offset = 1.0 / 1024.0;\n"
			"void main()\n"
			"{\n"
			"vec2 offsets[25] = vec2[](\n"
			"vec2(-2*offset,  2*offset),\n"
			"	vec2(-offset,  2*offset), \n"
			"	vec2( 0.0f,    2*offset), \n"
			"	vec2( offset,  2*offset), \n"
			"	vec2( 2*offset,  2*offset), \n"
			"	vec2(-2*offset,  offset), \n"
			"	vec2(-offset,  offset), \n"
			"	vec2( 0.0f,    offset), \n"
			"	vec2( offset,  offset),\n"
			"	vec2( 2*offset,  offset),\n"
			"	vec2(-2*offset,  0.0f), \n"
			"	vec2(-offset,  0.0f),\n"
			"	vec2( 0.0f,    0.0f), \n"
			"	vec2( offset,  0.0f),\n"
			"	vec2( 2*offset,  0.0f), \n"
			"	vec2(-2*offset,  -2*offset), \n"
			"	vec2(-offset,  -2*offset), \n"
			"	vec2( 0.0f,    -2*offset), \n"
			"	vec2( offset,  -2*offset), \n"
			"	vec2( 2*offset,  -2*offset), \n"
			"	vec2(-2*offset,  -offset), \n"
			"	vec2(-offset,  -offset), \n"
			"	vec2( 0.0f,    -offset), \n"
			"	vec2( offset,  -offset),\n"
			"	vec2( 2*offset,  -offset)\n"
			");\n"
			" vec3 sampleTex[25];\n"
			"	int num1=0;\n"
			"float f=0.0;\n"
			"for(int i = 0; i < 25; i++)\n"
			"{\n"
			"	sampleTex[i] = vec3(texture(screenTexture, TexCoords.st + offsets[i]));\n"
			"	if(sampleTex[i].z<0.9f){\n"
			"		{\n"
			"			num1++;\n"
			"			f=max(sampleTex[i].z,f);\n"
			"		}\n"
			"}\n"
			"vec3 col = vec3(0.0);\n"
			"if(num1>0&&sampleTex[12].x==1.0f){\n"
			"	FragColor = vec4(vec3(f), 1.0);\n"
			"}else{\n"
			"   FragColor = texture(screenTexture, TexCoords);\n"
			"}}}\0";
	// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
	// ---------------------------------------------------------------------------------------------------------
	void processInput(GLFWwindow *window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);
	}
	float uvTovectex(float uv) {
		if (uv == 0.5f) {
			return 0;
		}
		else if (uv>0.5f)
		{
			return 2 * (uv - 0.5f);
		}
		else
		{
			return 2 * uv - 1;
		}
	}
	std::ofstream myFile;
	void myOutput(unsigned char byte)
	{
		myFile << byte;
	}


}

int bake::modelToView(
	const std::string out_file,
	const uautil::Scene* scene,
	const size_t num_meshes,
	float** vertex_colors,
	float scene_bbox_min[3],
	float scene_bbox_max[3], std::vector<uautil::Mesh> blockers) {
	if(!out_file.empty()){
		OUT_FILE=out_file;
	}
	myFile.open(OUT_FILE);
	//std::cout<<"OUT_FILE:  "<<OUT_FILE<<"\n"<<std::endl;
	std::vector<float> vertices;
	std::vector<float> aoNumber;
	for (size_t j = 0; j <1; j++)//scene->m_num_meshes
	{
		auto mesh = scene->m_meshes[j];
		auto ao = vertex_colors[j];
		std::map<float, std::vector<float>> aoMap;
		for (size_t i = 0; i < scene->m_meshes[j].trianglesArray.size(); i++)
		{
			auto index = mesh.trianglesArray[i];

			vertices.push_back(uvTovectex(mesh.texcoordsArray[index.x].x));
			vertices.push_back(uvTovectex(mesh.texcoordsArray[index.x].y));
			vertices.push_back(0.0f);

			vertices.push_back(uvTovectex(mesh.texcoordsArray[index.y].x));
			vertices.push_back(uvTovectex(mesh.texcoordsArray[index.y].y));
			vertices.push_back(0.0f);

			vertices.push_back(uvTovectex(mesh.texcoordsArray[index.z].x));
			vertices.push_back(uvTovectex(mesh.texcoordsArray[index.z].y));
			vertices.push_back(0.0f);
			/*float scale = 0.02f;
			vertices.push_back(mesh.verticesArray[index.x].x/ scale);
			vertices.push_back(mesh.verticesArray[index.x].y / scale);
			vertices.push_back(mesh.verticesArray[index.x].z / scale);
			vertices.push_back(mesh.verticesArray[index.y].x / scale);
			vertices.push_back(mesh.verticesArray[index.y].y / scale);
			vertices.push_back(mesh.verticesArray[index.y].z / scale);
			vertices.push_back(mesh.verticesArray[index.z].x / scale);
			vertices.push_back(mesh.verticesArray[index.z].y / scale);
			vertices.push_back(mesh.verticesArray[index.z].z / scale);*/
			//std::cout << uvTovectex(std::abs(mesh.texcoordsArray[index.z].x)) << "::" << uvTovectex(std::abs(mesh.texcoordsArray[index.z].y)) << std::endl;
			aoNumber.push_back(vertex_colors[j][index.x]);
			aoNumber.push_back(vertex_colors[j][index.y]);
			
			aoNumber.push_back(vertex_colors[j][index.z]);


			/*if (aoMap.find(index.x) != aoMap.end()) {
				aoMap.find(index.x)->second.push_back(vertex_colors[j][index.x]);
			}
			else
			{
				std::vector<float> aoArray;
				aoArray.push_back(vertex_colors[j][index.x]);
				aoMap[index.x] = aoArray;
			}

			if (aoMap.find(index.y) != aoMap.end()) {
				aoMap.find(index.y)->second.push_back(vertex_colors[j][index.y]);
			}
			else
			{
				std::vector<float> aoArray;
				aoArray.push_back(vertex_colors[j][index.y]);
				aoMap[index.y] = aoArray;
			}

			if (aoMap.find(index.z) != aoMap.end()) {
				aoMap.find(index.z)->second.push_back(vertex_colors[j][index.z]);
			}
			else
			{
				std::vector<float> aoArray;
				aoArray.push_back(vertex_colors[j][index.z]);
				aoMap[index.z] = aoArray;
			}*/
		}

		//for (auto &it : aoMap) {
		//	//std::cout << it.first << "-" << std::flush;
		//	float count = 0.0f;
		//	for (auto a : it.second)
		//	{
		//		count += a;
		//	}
		//	auto num = count / it.second.size();
		//	it.second.clear();
		//	it.second.push_back(num);
		//	//std::cout << "  " << num << std::endl;
		//}

		//for (size_t i = 0; i < scene->m_meshes[j].trianglesArray.size(); i++) {
		//	auto index = mesh.trianglesArray[i];
		//	if (aoMap.find(index.x) != aoMap.end()) {
		//		aoNumber.push_back(aoMap.find(index.x)->second[0]);
		//	}
		//	if (aoMap.find(index.y) != aoMap.end()) {
		//		aoNumber.push_back(aoMap.find(index.y)->second[0]);
		//	}
		//	if (aoMap.find(index.z) != aoMap.end()) {
		//		aoNumber.push_back(aoMap.find(index.z)->second[0]);
		//	}
		//}

	}
	
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	GLFWwindow* window = glfwCreateWindow(WINDOWS_SIZE, WINDOWS_SIZE, "Baking.........", NULL, NULL);


	//
	//glEnable(GL_MULTISAMPLE);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);;

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
							 // positions   // texCoords
		-1.0f,  1.0f,  0.0f, 1.0f,
		-1.0f, -1.0f,  0.0f, 0.0f,
		1.0f, -1.0f,  1.0f, 0.0f,

		-1.0f,  1.0f,  0.0f, 1.0f,
		1.0f, -1.0f,  1.0f, 0.0f,
		1.0f,  1.0f,  1.0f, 1.0f
	};
	glEnable(GL_DEPTH_TEST);

	// build and compile our shader program
	// ------------------------------------
	// vertex shader
	int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	    glShaderSource(vertexShader, 1, &bakingAoShaderVS, NULL);
	    glCompileShader(vertexShader);
	    // check for shader compile errors
	    int success;
	    char infoLog[512];
	    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	    if (!success)
	    {
	        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
	        std::cout << "ERROR::SHADER::bakingAoShader::COMPILATION_FAILED\n" << infoLog << std::endl;
	    }
	    // fragment shader
	    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	    glShaderSource(fragmentShader, 1, &bakingAoShaderFS, NULL);
	    glCompileShader(fragmentShader);
	    // check for shader compile errors
	    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	    if (!success)
	    {
	        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
	        std::cout << "ERROR::SHADER::bakingAoShader::COMPILATION_FAILED\n" << infoLog << std::endl;
	    }
	    // link shaders
	    int shaderProgram = glCreateProgram();
	    glAttachShader(shaderProgram, vertexShader);
	    glAttachShader(shaderProgram, fragmentShader);
	    glLinkProgram(shaderProgram);
	    // check for linking errors
	    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	    if (!success) {
	        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
	        std::cout << "ERROR::SHADER::shaderProgram::LINKING_FAILED\n" << infoLog << std::endl;
	    }
	    glDeleteShader(vertexShader);
	    glDeleteShader(fragmentShader);
	// build and compile our shader program
	// ------------------------------------
	// vertex shader
	int vertexShader1 = glCreateShader(GL_VERTEX_SHADER);
	    glShaderSource(vertexShader1, 1, &bakingAoScreenVS, NULL);
	    glCompileShader(vertexShader1);
	    glGetShaderiv(vertexShader1, GL_COMPILE_STATUS, &success);
	    if (!success)
	    {
	        glGetShaderInfoLog(vertexShader1, 512, NULL, infoLog);
	        std::cout << "ERROR::SHADER::bakingAoScreen::vs::COMPILATION_FAILED\n" << infoLog << std::endl;
	    }
	    // fragment shader
	    int fragmentShader1 = glCreateShader(GL_FRAGMENT_SHADER);
	    glShaderSource(fragmentShader1, 1, &bakingAoScreenFS, NULL);
	    glCompileShader(fragmentShader1);
	    // check for shader compile errors
	    glGetShaderiv(fragmentShader1, GL_COMPILE_STATUS, &success);
	    if (!success)
	    {
	        glGetShaderInfoLog(fragmentShader1, 512, NULL, infoLog);
	        std::cout << "ERROR::SHADER::bakingAoScreen::fs::COMPILATION_FAILED\n" << infoLog << std::endl;
	    }
	    // link shaders
	    int shaderProgram1 = glCreateProgram();
	    glAttachShader(shaderProgram1, vertexShader1);
	    glAttachShader(shaderProgram1, fragmentShader1);
	    glLinkProgram(shaderProgram1);
	    // check for linking errors
	    glGetProgramiv(shaderProgram1, GL_LINK_STATUS, &success);
	    if (!success) {
	        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
	        std::cout << "ERROR::SHADER::shaderProgramScreen::LINKING_FAILED\n" << infoLog << std::endl;
	  }

	 glDeleteShader(vertexShader1);
	 glDeleteShader(fragmentShader1);
	 glUniform1i(glGetUniformLocation(shaderProgram1, "screenTexture"), 0);

	unsigned int  VAO;
	unsigned int VBO[2];

	glGenVertexArrays(1, &VAO);
	glGenBuffers(2, VBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, aoNumber.size() * sizeof(float), &aoNumber[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);

	////// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	////// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	////// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);

	// screen quad VAO
	unsigned int quadVAO, quadVBO;
	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));


	// framebuffer configuration
	// -------------------------
	unsigned int framebuffer, textureColorbuffer;
	glGenFramebuffers(1, &framebuffer);
	
	glGenTextures(1, &textureColorbuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOWS_SIZE, WINDOWS_SIZE, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);
	// create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
	unsigned int rbo;
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WINDOWS_SIZE, WINDOWS_SIZE); // use a single renderbuffer object for both a depth AND stencil buffer.
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
																								  // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// framebuffer configuration
	// -------------------------
	unsigned int framebuffer1, textureColorbuffer1;
	glGenFramebuffers(1, &framebuffer1);

	glGenTextures(1, &textureColorbuffer1);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WINDOWS_SIZE, WINDOWS_SIZE, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer1, 0);
	// create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
	unsigned int rbo1;
	glGenRenderbuffers(1, &rbo1);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo1);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WINDOWS_SIZE, WINDOWS_SIZE); // use a single renderbuffer object for both a depth AND stencil buffer.
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo1); // now actually attach it
																								  // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// uncomment this call to draw in wireframe polygons.
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	auto imgBuffer = (unsigned char*)malloc(WINDOWS_SIZE * WINDOWS_SIZE * 3);
	const bool isrgb = true;  // true = rgb image, else false = grayscale
	const int quality = 100;    // compression quality: 0 = worst, 100 = best, 80 to 90 are most often used
	const bool downsample = false; // false = save as ycbcr444 jpeg (better quality), true = ycbcr420 (smaller file)
	const char* comment = "uality ao image"; // arbitrary jpeg comment

		// render
		// ------
		glViewport(0, 0, WINDOWS_SIZE, WINDOWS_SIZE);
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
		glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
#if USE_NATIVE_OSMESA
		glfwGetOSMesaColorBuffer(window, &WINDOWS_SIZE, &WINDOWS_SIZE, NULL, (void **)&imgBuffer);
#else
		glReadPixels(0, 0, WINDOWS_SIZE, WINDOWS_SIZE, GL_RGB, GL_UNSIGNED_BYTE, imgBuffer);
#endif
		//glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, imgBuffer);
		auto ok = TooJpeg::writeJpeg(myOutput, imgBuffer, WINDOWS_SIZE, WINDOWS_SIZE, isrgb, quality, downsample, comment);
		glBindVertexArray(0); // no need to unbind it every time static_cast<GLsizei>(vertices.size())
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.

//		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);
//		glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessery actually, since we won't be able to see behind the quad anyways)
//		glClear(GL_COLOR_BUFFER_BIT);
		//glUseProgram(shaderProgram1);
//		glBindVertexArray(quadVAO);
//		glBindTexture(GL_TEXTURE_2D, textureColorbuffer);	// use the color attachment texture as the texture of the quad plane
//		glDrawArrays(GL_TRIANGLES, 0, 6);
//		glBindTexture(GL_TEXTURE_2D, textureColorbuffer1);
//		glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, imgBuffer);
//		//auto ok = TooJpeg::writeJpeg(myOutput, imgBuffer, WINDOWS_SIZE, WINDOWS_SIZE, isrgb, quality, downsample, comment);
//		glBindTexture(GL_TEXTURE_2D, 0);
//		glBindFramebuffer(GL_FRAMEBUFFER, 0);
//		glDisable(GL_DEPTH_TEST);
		if (ok) {
			delete[] imgBuffer;
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(2, VBO);
			glfwTerminate();
			return 0;
		}
		

	return 0;
}

