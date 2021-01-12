#pragma once
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstddef>
#include "optix/optixu/optixu_math_namespace.h"
#include "optix/optixu/optixu_matrix_namespace.h"
#include "scene.h"
// Note: cstdint would require building with c++11 on gcc
#if defined(_WIN32)
#include <cstdint>
#else
#include <stdint.h>
#endif

using namespace optix;

namespace bake
{


	struct SampleInfo
	{
		unsigned  tri_idx;
		float     bary[3];
		float     dA;
	};

	//------------------------------------------------------------------------------
	// A wrapper that provides more convenient return types
	/*class myMesh : public Mesh
	{
	public:
		int storage_identifier;
		float extra_samples_offset{};
		Matrix4x4 mat;
		
		float3  getBBoxMin() { return ptr_to_float3(bbox_min); }
		float3  getBBoxMax() { return ptr_to_float3(bbox_max); }
		int3*   getVertexIndices() { return reinterpret_cast<int3*>(tri_indices); }
		float3* getVertexData() { return reinterpret_cast<float3*>(positions); }

	private:
		float3 ptr_to_float3(const float* v) { return make_float3(v[0], v[1], v[2]); }
	};*/

	struct Instance
	{
		float xform[16];  // 4x4 row major
		uint64_t storage_identifier; // for saving the baked results
		unsigned mesh_index;
		float bbox_min[3];
		float bbox_max[3];
	};

	// Packages up geometry for routines below.
	// A scene is a set of instances that index into meshes. Meshes may be shared between instances.
	/*struct Scene
	{
		size_t num_meshes;
		std::vector<myMesh> receivers;
		float bbox_min[3]{ FLT_MAX, FLT_MAX, FLT_MAX };
		float bbox_max[3]{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	};*/


	struct AOSamples
	{
		size_t        num_samples;
		float*        sample_positions;
		float*        sample_normals;
		float*        sample_face_normals;
		SampleInfo*   sample_infos;
	};

	enum VertexFilterMode
	{
		VERTEX_FILTER_AREA_BASED = 0,
		VERTEX_FILTER_LEAST_SQUARES,
		VERTEX_FILTER_INVALID
	};

	void map_ao_to_vertices(
		const uautil::Scene&            scene,
		const size_t*           num_samples_per_instance,
		const AOSamples&        ao_samples,
		const float*            ao_values,
		const VertexFilterMode  mode,
		const float             regularization_weight,
		float**                 vertex_ao
	);
}
