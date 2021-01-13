/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <cassert>
#include <vector>
#include "optix/optixu/optixu_math_namespace.h"

#include "bake_api.h"
#include "bake_sample.h"
#include "bake_sample_internal.h"
#include "include/cuda/random.h"

using namespace optix;

// Ref: https://en.wikipedia.org/wiki/Halton_sequence
template <unsigned int TBase>
float halton(const unsigned int index)
{
	auto result = 0.0f;
	const auto inv_base = 1.0f / TBase;
	auto f = inv_base;
	auto i = index;
	while (i > 0)
	{
		result += f * (i % TBase);
		i = i / TBase;
		f *= inv_base;
	}
	return result;
}

float3 faceforward(const float3& normal, const float3& geom_normal)
{
	if (dot(normal, geom_normal) > 0.0f) return normal;
	return -normal;
}

float3 operator*(const optix::Matrix4x4& mat, const float3& v)
{
	return make_float3(mat * make_float4(v, 1.0f));
}

struct Angle
{
	std::size_t order;//˳��
	std::size_t tri_idx;
};

void sample_triangle(const optix::Matrix4x4& xform, const optix::Matrix4x4& xform_invtrans, const float3** verts, const float3** normals,
	const size_t tri_idx, const size_t tri_sample_count, const double tri_area, const unsigned base_seed, const float extra_offset,
	float3* sample_positions, float3* sample_norms, float3* sample_face_norms, bake::SampleInfo* sample_infos)
{
	const auto& v0 = *verts[0];
	const auto& v1 = *verts[1];
	const auto& v2 = *verts[2];

	//如果normals不存在就给一个默认的normal
	const auto face_normal = !normals 
		? float3{ 0.f, 1.f, 0.f }
	: normalize(cross(v1 - v0, v2 - v0));
	//const auto face_normal= normalize(cross(v1 - v0, v2 - v0));
	//std::cout << face_normal.x << "   " <<face_normal.y <<"    " << face_normal.z << std::endl;
	float3 n0, n1, n2;
	if (normals)
	{
		n0 = faceforward(*normals[0], face_normal);
		n1 = faceforward(*normals[1], face_normal);
		n2 = faceforward(*normals[2], face_normal);
	}
	else
	{
		// missing vertex normals, so use face normal.
		n0 = face_normal;
		n1 = face_normal;
		n2 = face_normal;
	}
	// std::cout << n0.x << "   " <<n0.y <<"    " << n0.z << std::endl;
	// std::cout << n1.x << "   " <<n1.y <<"    " << n1.z << std::endl;
	// std::cout << n2.x << "   " <<n2.y <<"    " << n2.z <<"\n"<< std::endl;
	// Random offset per triangle, to shift Halton points
	auto seed = tea<4>(base_seed, unsigned(tri_idx));
	const auto offset = make_float2(rnd(seed), rnd(seed));

	for (size_t index = 0; index < tri_sample_count; ++index)
	{
		sample_infos[index].tri_idx = unsigned(tri_idx);
		sample_infos[index].dA = static_cast<float>(tri_area / tri_sample_count);

		// Random point in unit square
		auto r1 = offset.x + halton<2>(unsigned(index) + 1);
		r1 = r1 - int(r1);
		auto r2 = offset.y + halton<3>(unsigned(index) + 1);
		r2 = r2 - int(r2);
		assert(r1 >= 0 && r1 <= 1);
		assert(r2 >= 0 && r2 <= 1);

		// Map to triangle. Ref: PBRT 2nd edition, section 13.6.4
		float3& bary = *reinterpret_cast<float3*>(sample_infos[index].bary);
		const float sqrt_r1 = sqrt(r1);
		bary.x = 1.0f - sqrt_r1;
		bary.y = r2 * sqrt_r1;
		bary.z = 1.0f - bary.x - bary.y;

		sample_norms[index] = normalize(xform_invtrans * (bary.x * n0 + bary.y * n1 + bary.z * n2));
		sample_face_norms[index] = normalize(xform_invtrans * face_normal);
		//magic number
		sample_positions[index] = xform * (bary.x * v0 + bary.y * v1 + bary.z * v2); //+ sample_face_norms[index] *0.6;
	}
}

double triangle_area(const float3& v0, const float3& v1, const float3& v2)
{
	const auto c = cross(v1 - v0, v2 - v0);
	return 0.5 * sqrt(c.x * c.x + c.y * c.y + c.z * c.z);
}

class TriangleSamplerCallback
{
public:
	TriangleSamplerCallback(const unsigned int min_samples_per_triangle, const double* area_per_triangle)
		: min_samples_per_triangle_(min_samples_per_triangle), area_per_triangle_(area_per_triangle) {}

	unsigned int min_samples(size_t i) const
	{
		return min_samples_per_triangle_; // same for every triangle
	}

	double area(size_t i) const
	{
		return area_per_triangle_[i];
	}

private:
	const unsigned int min_samples_per_triangle_;
	const double* area_per_triangle_;
};


const float3* get_vertex(const float* v, const unsigned stride_bytes, const int index)
{
	return reinterpret_cast<const float3*>(reinterpret_cast<const unsigned char*>(v) + index * stride_bytes);
}

const float get_cos(const size_t order, const float3& v0, const float3& v1, const float3& v2) {
	float angle_cos = 0.0f;
	switch (order)
	{
	case 0:

		angle_cos =std::acosf( dot(normalize( v1 - v0), normalize( v2 - v0)));
		break;
	case 1:
		angle_cos = std::acosf(dot(normalize(v0 - v1), normalize(v2 - v1)));
		break;
	case 2:
		angle_cos = std::acosf(dot(normalize(v0 - v2), normalize(v1 - v2)));
		break;
	default:
		break;
	}
	return angle_cos;
}

void sample_instance(
	const uautil::Mesh* mesh,
	const optix::Matrix4x4& xform,
	const unsigned int seed,
	const size_t min_samples_per_triangle,
	bake::AOSamples& ao_samples)
{
	// Setup access to mesh data
	const auto xform_invtrans = xform.inverse().transpose();
	assert(ao_samples.num_samples >= mesh->num_triangles*min_samples_per_triangle);
	assert(mesh->num_vertices);
	assert(ao_samples.sample_positions);
	assert(ao_samples.sample_normals);
	assert(ao_samples.sample_infos);

	const int3* tri_vertex_indices = reinterpret_cast<const int3*>(&mesh->trianglesArray[0]);
	const auto sample_positions = reinterpret_cast<float3*>(ao_samples.sample_positions);
	const auto sample_norms = reinterpret_cast<float3*>(ao_samples.sample_normals);
	const auto sample_face_norms = reinterpret_cast<float3*>(ao_samples.sample_face_normals);
	const auto sample_infos = ao_samples.sample_infos;

	// Compute triangle areas
	std::vector<double> tri_areas(mesh->trianglesArray.size(), 0.0);
	for (size_t tri_idx = 0; tri_idx < mesh->trianglesArray.size(); tri_idx++)
	{
		const auto& tri = tri_vertex_indices[tri_idx];
		const float3* verts[] = {
			(float3*)&mesh->verticesArray[tri.x],
			(float3*)&mesh->verticesArray[tri.y],
			(float3*)&mesh->verticesArray[tri.z]
		};
		const auto area = triangle_area(xform * verts[0][0], xform * verts[1][0], xform * verts[2][0]);
		tri_areas[tri_idx] = area;
	}

	// Get sample counts
	std::vector<size_t> tri_sample_counts(mesh->trianglesArray.size(), 0);
	const TriangleSamplerCallback cb(unsigned(min_samples_per_triangle), &tri_areas[0]);
	distribute_samples_generic(cb, ao_samples.num_samples, mesh->trianglesArray.size(), &tri_sample_counts[0]);

	// Place samples
	size_t sample_idx = 0;
	for (size_t tri_idx = 0; tri_idx < mesh->trianglesArray.size(); tri_idx++)
	{
		const int3& tri = tri_vertex_indices[tri_idx];
		const float3* verts[] = {
			(float3*)&mesh->verticesArray[tri.x],
			(float3*)&mesh->verticesArray[tri.y],
			(float3*)&mesh->verticesArray[tri.z]
		};
		const float3** normals = nullptr;
		const float3* norms[3];
		if (mesh->normalsArray.size()!=0)
		{
			norms[0] = (float3*)&mesh->normalsArray[tri.x];
			norms[1] = (float3*)&mesh->normalsArray[tri.y];
			norms[2] = (float3*)&mesh->normalsArray[tri.z];
			normals = norms;
		}
		sample_triangle(xform, xform_invtrans, verts, normals, tri_idx, tri_sample_counts[tri_idx], tri_areas[tri_idx], seed, mesh->extra_samples_offset, 
			sample_positions + sample_idx, sample_norms + sample_idx, sample_face_norms + sample_idx, sample_infos + sample_idx);
		sample_idx += tri_sample_counts[tri_idx];
	}

	assert(sample_idx == ao_samples.num_samples);
}


void bake::sample_instances(const uautil::Scene& scene, const size_t* num_samples_per_instance, const size_t min_samples_per_triangle,
	AOSamples& ao_samples)
{
	std::vector<size_t> sample_offsets(scene.m_num_meshes);
	{
		size_t sample_offset = 0;
		for (size_t i = 0; i < scene.m_num_meshes; ++i)
		{
			sample_offsets[i] = sample_offset;
			sample_offset += num_samples_per_instance[i];
		}
	}


	#pragma omp parallel for
	for (ptrdiff_t i = 0; i < ptrdiff_t(scene.m_num_meshes); ++i)
	{
		const auto sample_offset = sample_offsets[i];

		// Point to samples for this instance
		AOSamples instance_ao_samples{};
		instance_ao_samples.num_samples = num_samples_per_instance[i];
		instance_ao_samples.sample_positions = ao_samples.sample_positions + 3 * sample_offset;
		instance_ao_samples.sample_normals = ao_samples.sample_normals + 3 * sample_offset;
		instance_ao_samples.sample_face_normals = ao_samples.sample_face_normals + 3 * sample_offset;
		instance_ao_samples.sample_infos = ao_samples.sample_infos + sample_offset;

		optix::Matrix4x4 xform(scene.m_meshes[i].transform);
		sample_instance(&scene.m_meshes[i], xform, unsigned(i), min_samples_per_triangle, instance_ao_samples);
	}
}

class InstanceSamplerCallback
{
public:
	InstanceSamplerCallback(const unsigned int* min_samples_per_instance, const double* area_per_instance)
		: min_samples_per_instance_(min_samples_per_instance), area_per_instance_(area_per_instance) {}

	unsigned int min_samples(const size_t i) const
	{
		return min_samples_per_instance_[i];
	}

	double area(const size_t i) const
	{
		return area_per_instance_[i];
	}

private:
	const unsigned int* min_samples_per_instance_;
	const double* area_per_instance_;
};

size_t bake::distribute_samples(
	const uautil::Scene& scene,
	const size_t min_samples_per_triangle,
	const size_t requested_num_samples,
	size_t* num_samples_per_instance)
{
	// Compute min samples per instance
	std::vector<unsigned int> min_samples_per_instance(scene.m_num_meshes);
	size_t num_triangles = 0;
	for (size_t i = 0; i < scene.m_num_meshes; ++i)
	{
		const auto& mesh = scene.m_meshes[i];
		min_samples_per_instance[i] = unsigned(min_samples_per_triangle * mesh.num_triangles);
		num_triangles += mesh.num_triangles;
	}
	const auto min_num_samples = min_samples_per_triangle * num_triangles;
	const auto num_samples = std::max(min_num_samples, requested_num_samples);

	// Compute surface area per instance.
	// Note: for many xforms, we could compute surface area per mesh instead of per instance.
	std::vector<double> area_per_instance(scene.m_num_meshes, 0.0);
	if (num_samples > min_num_samples)
	{
		#pragma omp parallel for
		for (ptrdiff_t idx = 0; idx < ptrdiff_t(scene.m_num_meshes); ++idx)
		{
			const auto& mesh = scene.m_meshes[idx];
			std::cout<<"samples 大于 默认值 \n"<<std::endl;
			//const Matrix4x4 xform(scene.receivers[idx]->matrix._elems);
			/*const int3* tri_vertex_indices = reinterpret_cast<const int3*>(&mesh->triangles[0]);
			const unsigned vertex_stride_bytes = 3 * sizeof(float);
			for (size_t tri_idx = 0; tri_idx < mesh->triangles.size(); ++tri_idx)
			{
				const auto& tri = tri_vertex_indices[tri_idx];
				const float3* verts[] = {
					(float3*)&mesh->vertices[tri.x],
					(float3*)&mesh->vertices[tri.y],
					(float3*)&mesh->vertices[tri.z]
				};	
				xform*verts[0];
				const auto area = triangle_area(xform * verts[0][0], xform * verts[1][0], xform * verts[2][0]);
				area_per_instance[idx] += area;
			}*/
		}
	}

	// Distribute samples
	const InstanceSamplerCallback cb(&min_samples_per_instance[0], &area_per_instance[0]);
	distribute_samples_generic(cb, num_samples, scene.m_num_meshes, num_samples_per_instance);

	return num_samples;
}
