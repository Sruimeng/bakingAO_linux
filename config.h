#pragma once
#include <iostream>
#include "bake_api.h"
namespace
{
	const size_t NUM_RAYS = 64;
	const size_t SAMPLES_PER_FACE = 3;
} // namespace

namespace bake
{
	struct Config
	{
		std::string scene_filename;
		size_t num_instances_per_mesh;
		int num_samples;
		int min_samples_per_face;
		int num_rays;

		bake::VertexFilterMode filter_mode;
		float regularization_weight;
		bool use_ground_plane_blocker;
		bool use_viewer;
		int ground_upaxis;
		float ground_scale_factor;
		float ground_offset_factor;
		int32_t scene_offset_scale;
		float scene_maxdistance_scale;
		float scene_maxdistance;
		int32_t scene_offset;
		std::string output_filename;
		int uv_offset = 1;
		int img_single = 1;
		int normal_fix = 0;
		Config(int argc, const char **argv)
		{
			// set defaults
			num_instances_per_mesh = 1;
			num_samples = 0; // default means determine from mesh

			min_samples_per_face = SAMPLES_PER_FACE;
			num_rays = NUM_RAYS;
			ground_upaxis = 1;
			ground_scale_factor = 10.0f;
			ground_offset_factor = 0.03f;
			scene_offset_scale = 3;
			scene_maxdistance_scale = 10.f;
			regularization_weight = 0.1f;
			scene_offset = 1; // must default to 0
			scene_maxdistance = 0;
			use_ground_plane_blocker = true;
			use_viewer = true;

			//VERTEX_FILTER_LEAST_SQUARES VERTEX_FILTER_AREA_BASED
			filter_mode = bake::VERTEX_FILTER_AREA_BASED;
			// parse arguments
			for (int i = 1; i < argc; ++i)
			{
				std::string arg(argv[i]);
				if (arg.empty())
					continue;

				if (arg == "-h" || arg == "--help")
				{
					printUsageAndExit(argv[0]);
				}
				else if ((arg == "-i" || arg == "--infile") && i + 1 < argc)
				{
					scene_filename = argv[++i];
				}
				else if ((arg == "-o" || arg == "--outfile") && i + 1 < argc)
				{
					output_filename = argv[++i];
					//std::cout<<"output_filename:  "<<output_filename<<"\n"<<std::endl;
				}
				else if ((arg == "-d" || arg == "--distance") && i + 1 < argc)
				{
					scene_offset = atoi(argv[++i]);
					//std::cout<<"scene_offset_scale:  "<<scene_offset_scale<<"\n"<<std::endl;
				}
				else if ((arg == "-s" || arg == "--scale") && i + 1 < argc)
				{
					scene_offset_scale = atoi(argv[++i]);
					//std::cout<<"scene_offset_scale:  "<<scene_offset_scale<<"\n"<<std::endl;
				}
				else if ((arg == "-m" || arg == "--max") && i + 1 < argc)
				{
					scene_maxdistance_scale = atof(argv[++i]);
					//std::cout<<"scene_maxdistance_scale:  "<<scene_maxdistance_scale<<"\n"<<std::endl;
				}
				else if ((arg == "-u" || arg == "--uvoffset") && i + 1 < argc)
				{
					uv_offset = atoi(argv[++i]);
					//std::cout<<"scene_maxdistance_scale:  "<<scene_maxdistance_scale<<"\n"<<std::endl;
				}
				else if ((arg == "-f" || arg == "--sigle_img_flag") && i + 1 < argc)
				{
					img_single = atoi(argv[++i]);
					//std::cout<<"scene_maxdistance_scale:  "<<scene_maxdistance_scale<<"\n"<<std::endl;
				}
				else if ((arg == "-n" || arg == "--fix_normals") && i + 1 < argc)
				{
					normal_fix = atoi(argv[++i]);
					//std::cout<<"scene_maxdistance_scale:  "<<scene_maxdistance_scale<<"\n"<<std::endl;
				}
			}
		}

		void printUsageAndExit(const char *argv0)
		{
			std::cerr
				<< "Usage  : " << argv0 << " [options]\n"
				<< "App options:\n"
				<< "  -h  | --help	帮助信息\n"
				<< "  -i  | --infile <model_file(string)>	输入文件的地址\n"
				<< "  -o  | --outfile <image_file(string)>	输出文件的地址\n"
				<< "  -d  | --distance <offset(number)>	offset的数值\n"
				<< "  -s | --scale <model(number)> offset的层数\n"
				<< "  -m  | --max <max_distance(muber)>	射线的的最大距离\n"
				<< "  -u  | --uvoffset <uvoffset(number)> 接缝像素的偏移\n"
				<< "  -f  | --sigle_img_flag <sigle_img_flag(number)> 是否输出为一张图片\n"
				<< "  -n  | --fix_normals <fix_normals_flag(number)> 是否修复normals\n"
				<< "........可能以后有其他的属性\n"
				<< std::endl;

			exit(1);
		}
	};
} // namespace bake
