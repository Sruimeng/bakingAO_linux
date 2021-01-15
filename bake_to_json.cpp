#include "scene.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "include/tinygltf/json.hpp"
using json = nlohmann::json;
void bakeToJson(const uautil::Scene* scene, const std::map<int32_t, std::string>* t_img_name_map,const int img_sigle_) 
{
	//新建输出json对象
	json out_json;
	for (size_t i = 0; i < scene->m_meshes.size(); i++)
	{
		uautil::Mesh mesh = scene->m_meshes[i];
		
		json j;
		j["meshIndex"] = mesh.mesh_idx;
		j["primitiveIndex"] = mesh.primitive_idx;
		if (img_sigle_) {
			j["aoTexture"] = t_img_name_map->at(0);
		}
		else
		{
			std::string f_image_name = t_img_name_map->at(mesh.image_name);
			j["aoTexture"] = f_image_name;
		}
		
		out_json["mapping"].push_back(j);
	}
	std::string s = "out_image.json";
	std::ofstream outFile(s);
	outFile << std::setw(4) << out_json << std::endl;
};