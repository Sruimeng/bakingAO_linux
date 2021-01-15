
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
		const bake::Config config(argc, argv);
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

		float aabb_length = sqrtf((scene.bbox_max[0] - scene.bbox_min[0]) * (scene.bbox_max[0] - scene.bbox_min[0]) + (scene.bbox_max[1] - scene.bbox_min[1]) * (scene.bbox_max[1] - scene.bbox_min[1]) + (scene.bbox_max[2] - scene.bbox_min[2]) * (scene.bbox_max[2] - scene.bbox_min[2]));
		//std::cerr << "aabb_length: " << aabb_length << "\n";
		float aabb_scale = pow(0.1, 3);//0.1����
		float final_length = aabb_length;
		int pow_number = 1;
		while (final_length > 1.0)
		{
			final_length = aabb_length * pow(0.1, pow_number);
			pow_number++;
		}
		float final_scale = pow(0.1, config.scene_offset_scale+ pow_number - 1)*config.scene_offset;
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
			&config, &num_samples_per_instance[0]);
		allocate_ao_samples(ao_samples, total_samples);
		sample_instances(scene, &num_samples_per_instance[0], &config, ao_samples);
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
			scene_offset = scene_scale * final_scale;
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
		bake::modelToView(&config,&scene, scene.m_num_meshes, vertex_ao, scene.bbox_min, scene.bbox_max, blocker_meshes);
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
