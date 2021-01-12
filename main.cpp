
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <iostream>

#include "optix/optix_prime/optix_primepp.h"
#include "optix/optixu/optixu_math_namespace.h"
#include "optix/optixu/optixu_matrix_namespace.h"

#include "bake_api.h"
#include "bake_util.h"
#include "bake_sample.h"
#include "bake_ao_optix_prime.h"
#include "bake_to_image.h"
#include "toojpeg.h"
#include "scene.h"
using namespace optix::prime;
using optix::fmaxf;

namespace
{

	float3 make_float3(const float *a)
	{
		return ::make_float3(a[0], a[1], a[2]);
	}
	void set_vertex_entry(float *vertices, const int idx, const int axis, float *vec)
	{
		vertices[3 * idx + axis] = vec[axis];
	}

	inline void expand_bbox(float bbox_min[3], float bbox_max[3], const float v[3])
	{
		for (size_t k = 0; k < 3; ++k)
		{
			bbox_min[k] = std::min(bbox_min[k], v[k]);
			bbox_max[k] = std::max(bbox_max[k], v[k]);
		}
	}
	void allocate_ao_samples(bake::AOSamples &ao_samples, size_t n)
	{
		ao_samples.num_samples = n;
		ao_samples.sample_positions = new float[3 * n];
		ao_samples.sample_normals = new float[3 * n];
		ao_samples.sample_face_normals = new float[3 * n];
		ao_samples.sample_infos = new bake::SampleInfo[n];
	}
	void destroy_ao_samples(bake::AOSamples &ao_samples)
	{
		delete[] ao_samples.sample_positions;
		ao_samples.sample_positions = nullptr;
		delete[] ao_samples.sample_normals;
		ao_samples.sample_normals = nullptr;
		delete[] ao_samples.sample_face_normals;
		ao_samples.sample_face_normals = nullptr;
		delete[] ao_samples.sample_infos;
		ao_samples.sample_infos = nullptr;
		ao_samples.num_samples = 0;
	}
	void make_ground_plane(const float scene_bbox_min[3], const float scene_bbox_max[3], int upaxis, const float scale_factor,
						   const float offset_factor, std::vector<uautil::Mesh> &meshes, uautil::Scene &scene)
	{
		auto plane_mesh = new uautil::Mesh();
		plane_mesh->verticesArray.resize(4);
		plane_mesh->trianglesArray = {
			{0, 1, 2},
			{0, 2, 3},
			{2, 1, 0},
			{3, 2, 0}};

		float scene_extents[] = {
			scene_bbox_max[0] - scene_bbox_min[0],
			scene_bbox_max[1] - scene_bbox_min[1],
			scene_bbox_max[2] - scene_bbox_min[2]};

		float ground_min[] = {
			scene_bbox_max[0] - scale_factor * scene_extents[0],
			scene_bbox_min[1] - scale_factor * scene_extents[1],
			scene_bbox_max[2] - scale_factor * scene_extents[2]};
		float ground_max[] = {
			scene_bbox_min[0] + scale_factor * scene_extents[0],
			scene_bbox_min[1] + scale_factor * scene_extents[1],
			scene_bbox_min[2] + scale_factor * scene_extents[2]};

		if (upaxis > 2)
		{
			upaxis %= 3;
			ground_min[upaxis] = scene_bbox_max[upaxis] + scene_extents[upaxis] * offset_factor;
			ground_max[upaxis] = scene_bbox_max[upaxis] + scene_extents[upaxis] * offset_factor;
		}
		else
		{
			ground_min[upaxis] = scene_bbox_min[upaxis] - scene_extents[upaxis] * offset_factor;
			ground_max[upaxis] = scene_bbox_min[upaxis] - scene_extents[upaxis] * offset_factor;
		}

		const auto axis0 = (upaxis + 2) % 3;
		const auto axis1 = (upaxis + 1) % 3;

		float vertex_data[4 * 3] = {};
		set_vertex_entry(vertex_data, 0, upaxis, ground_min);
		set_vertex_entry(vertex_data, 0, axis0, ground_min);
		set_vertex_entry(vertex_data, 0, axis1, ground_min);

		set_vertex_entry(vertex_data, 1, upaxis, ground_min);
		set_vertex_entry(vertex_data, 1, axis0, ground_max);
		set_vertex_entry(vertex_data, 1, axis1, ground_min);

		set_vertex_entry(vertex_data, 2, upaxis, ground_min);
		set_vertex_entry(vertex_data, 2, axis0, ground_max);
		set_vertex_entry(vertex_data, 2, axis1, ground_max);

		set_vertex_entry(vertex_data, 3, upaxis, ground_min);
		set_vertex_entry(vertex_data, 3, axis0, ground_min);
		set_vertex_entry(vertex_data, 3, axis1, ground_max);

		for (size_t i = 0; i < 4; ++i)
		{
			plane_mesh->verticesArray[i].x = vertex_data[3 * i];
			plane_mesh->verticesArray[i].y = vertex_data[3 * i + 1];
			plane_mesh->verticesArray[i].z = vertex_data[3 * i + 2];
		}
		plane_mesh->num_vertices = 4;
		plane_mesh->num_triangles = 4;
		plane_mesh->transform.identity();
		expand_bbox(scene.bbox_min, scene.bbox_max, ground_min);
		expand_bbox(scene.bbox_min, scene.bbox_max, ground_max);
		for (size_t k = 0; k < 3; ++k)
		{
			plane_mesh->bbox_min[k] = ground_min[k];
			plane_mesh->bbox_max[k] = ground_max[k];
		}
		//plane_mesh->normalsArray = nullptr;
		meshes.push_back(*plane_mesh);
	}

} // namespace

const size_t NUM_RAYS = 64;
const size_t SAMPLES_PER_FACE = 3;

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
	float scene_offset_scale;
	float scene_maxdistance_scale;
	float scene_maxdistance;
	float scene_offset;
	std::string output_filename;

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
		scene_offset_scale = .001f;
		scene_maxdistance_scale = 10.f;
		regularization_weight = 0.1f;
		scene_offset = 0; // must default to 0
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
				scene_offset_scale = atof(argv[++i]);
				//std::cout<<"scene_offset_scale:  "<<scene_offset_scale<<"\n"<<std::endl;
			}
			else if ((arg == "-m" || arg == "--max") && i + 1 < argc)
			{
				scene_maxdistance_scale = atof(argv[++i]);
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
			<< "  -m  | --max <max_distance(muber)>	射线的的最大距离\n"
			<< "........可能以后有其他的属性\n"
			<< std::endl;

		exit(1);
	}
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
	// set defaults
	RTPcontexttype contextType = RTP_CONTEXT_TYPE_CUDA;
	RTPbuffertype bufferType = RTP_BUFFER_TYPE_HOST;
	//	const std::string filename = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.5.0/SDK";
	//	///data/LeePerrySmith.glb rbhooh10_nbc//���� 02zmjnwc_w2x 10-7//���� ekplogdm_lyl//����
	//	std::string gltfFilename ="../02zmjnwc_w2x.glb";
	try
	{
		//
		// 默认参数设置
		//
		const Config config(argc, argv);
		//
		// 计时工具
		//
		Timer timer;
		timer.start();
		//
		// Load scene
		//
		std::cerr << "Load scene ...              ";
		std::cerr.flush();

		//创建场景
		uautil::Scene scene;
		uautil::loadScene(config.scene_filename, &scene);

		printTimeElapsed(timer);
		// Print scene stats
		{
			std::cerr << "Loaded scene: " << config.scene_filename << std::endl;
			std::cerr << "\t" << scene.m_num_meshes << " meshes, " << std::endl;
			size_t num_vertices = 0;
			size_t num_triangles = 0;
			for (size_t i = 0; i < scene.m_num_meshes; ++i)
			{
				num_vertices += scene.m_meshes[i].num_vertices;
				num_triangles += scene.m_meshes[i].num_triangles;
			}
			std::cerr << "\tuninstanced vertices: " << num_vertices << std::endl;
			std::cerr << "\tuninstanced triangles: " << num_triangles << std::endl;
		}

		//
		// Generate AO samples
		//

		std::cerr << "Minimum samples per face: " << config.min_samples_per_face << std::endl;

		std::cerr << "Generate sample points ... \n";
		std::cerr.flush();

		timer.reset();
		timer.start();

		std::vector<size_t> num_samples_per_instance(scene.m_num_meshes);
		bake::AOSamples ao_samples{};
		size_t total_samples = bake::distribute_samples(scene,
														config.min_samples_per_face, config.num_samples, &num_samples_per_instance[0]);
		allocate_ao_samples(ao_samples, total_samples);
		sample_instances(scene, &num_samples_per_instance[0], config.min_samples_per_face, ao_samples);
		printTimeElapsed(timer);
		std::cout << "\tTotal samples: " << total_samples << std::endl;

		//
		// Evaluate AO samples
		//
		std::cerr << "Compute AO ...             ";
		std::cerr.flush();

		timer.reset();
		timer.start();

		float scene_maxdistance;
		float scene_offset;
		{
			const float scene_scale = std::max(std::max(
												   scene.bbox_max[0] - scene.bbox_min[0],
												   scene.bbox_max[1] - scene.bbox_min[1]),
											   scene.bbox_max[2] - scene.bbox_min[2]);
			scene_maxdistance = scene_scale * config.scene_maxdistance_scale;
			scene_offset = scene_scale * config.scene_offset_scale;
			if (config.scene_offset)
			{
				scene_offset = config.scene_offset;
			}
			if (config.scene_maxdistance)
			{
				scene_maxdistance = config.scene_maxdistance;
			}
		}

		std::vector<float> ao_values(total_samples);
		std::vector<uautil::Mesh> blocker_meshes;
		{
			std::fill(ao_values.begin(), ao_values.end(), 0.0f);
			if (config.use_ground_plane_blocker)
			{
				make_ground_plane(scene.bbox_min, scene.bbox_max, config.ground_upaxis, config.ground_scale_factor, config.ground_offset_factor, blocker_meshes, scene);
			}

			ao_optix_prime(blocker_meshes, ao_samples,
						   config.num_rays, scene_offset, scene_maxdistance,
						   scene.bbox_min, scene.bbox_max, ao_values, scene);
		}
		std::cerr << "Map AO to vertices  ...    ";
		std::cerr.flush();

		timer.reset();
		timer.start();
		// Mapping AO to vertices
		auto vertex_ao = new float *[scene.m_num_meshes];
		{
			for (size_t i = 0U; i < scene.m_num_meshes; i++)
			{
				vertex_ao[i] = new float[scene.m_meshes[i].verticesArray.size()];
			}

			map_ao_to_vertices(scene, &num_samples_per_instance[0], ao_samples, &ao_values[0], config.filter_mode, config.regularization_weight, vertex_ao);
		}

		printTimeElapsed(timer);
		timer.reset();
		timer.start();
		std::cerr << "Save vertex ao ...              ";
		std::cerr.flush();
		bake::modelToView(config.output_filename, &scene, scene.m_num_meshes, vertex_ao, scene.bbox_min, scene.bbox_max, blocker_meshes);
		printTimeElapsed(timer);

		// Releasing some crap
		for (size_t i = 0; i < scene.m_num_meshes; ++i)
		{
			delete[] vertex_ao[i];
		}
		delete[] vertex_ao;
		destroy_ao_samples(ao_samples);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		exit(1);
	}
}
