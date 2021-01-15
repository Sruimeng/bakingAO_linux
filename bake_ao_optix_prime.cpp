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



#include "optix/optix_prime/optix_primepp.h"
//#include <optixu\optixu_math_namespace.h>
//#include <optixu\optixu_matrix_namespace.h>
#include <algorithm>
#include <map>
#include <fstream>
#include <iostream>
#include <chrono>
#include "include/putil/Buffer.h"
#include "toojpeg.h"
#include "bake_ao_optix_prime.h"
#include "bake_kernels.h"
#include "bake_util.h"

using namespace optix;
using namespace prime;
#define ACCUM_TIME( t, x )        \
do {                              \
  t.start();                      \
  x;                              \
  t.stop();                       \
} while( false )
namespace
{
	void create_instances(Context& context,
		std::vector<uautil::Mesh> meshes,
		const bool use_cuda,
		// output, to keep allocations around
		std::vector<Buffer<float3>*>& allocated_vertex_buffers, std::vector<Buffer<int3>*>& allocated_index_buffers,
		std::vector<Model>& models, std::vector<RTPmodel>& prime_instances, std::vector<optix::Matrix4x4>& transforms)
	{
		const auto buffer_type = use_cuda ? RTP_BUFFER_TYPE_CUDA_LINEAR : RTP_BUFFER_TYPE_HOST;
		const auto copy_type = use_cuda ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;

		// For sharing identical buffers between Models
		std::map< float*, Buffer<float3>* > unique_vertex_buffers;
		std::map< int*, Buffer<int3>* > unique_index_buffers;

		std::vector<float> vetix;
		const auto model_offset = models.size();
		models.reserve(models.size() + meshes.size());
		for (auto mesh : meshes)
		{
			auto model = context->createModel();
			for (size_t i = 0; i < mesh.num_vertices; i++)
			{
				vetix.push_back(mesh.verticesArray[i].x);
				vetix.push_back(mesh.verticesArray[i].y);
				vetix.push_back(mesh.verticesArray[i].z);
			}
			// Allocate or reuse vertex buffer and index buffer
			Buffer<float3>* vertex_buffer = NULL;
			if (unique_vertex_buffers.find(&vetix[0]) != unique_vertex_buffers.end())
			{
				vertex_buffer = unique_vertex_buffers.find(&vetix[0])->second;
			}
			else
			{
				// Note: copy disabled for Buffer, so need pointer here
				vertex_buffer = new Buffer<float3>(mesh.num_vertices, buffer_type, UNLOCKED);
				cudaMemcpy(vertex_buffer->ptr(), &vetix[0], vertex_buffer->sizeInBytes(), copy_type);
				auto b = vertex_buffer->sizeInBytes();
				auto a = vertex_buffer->hostPtr();
				unique_vertex_buffers[&vetix[0]] = vertex_buffer;
				allocated_vertex_buffers.push_back(vertex_buffer);
			}

			Buffer<int3>* index_buffer = nullptr;
			if (unique_index_buffers.find(&mesh.trianglesArray[0].x) != unique_index_buffers.end())
			{
				index_buffer = unique_index_buffers.find(&mesh.trianglesArray[0].x)->second;
			}
			else
			{
				index_buffer = new Buffer<int3>(mesh.num_triangles, buffer_type);
				cudaMemcpy(index_buffer->ptr(), &mesh.trianglesArray[0].x, index_buffer->sizeInBytes(), copy_type);
				unique_index_buffers[&mesh.trianglesArray[0].x] = index_buffer;
				auto a = index_buffer->hostPtr();
				allocated_index_buffers.push_back(index_buffer);
			}
			/*models->setTriangles(mesh.num_triangles, RTP_BUFFER_TYPE_HOST, &mesh.tri_indices,
				mesh.num_vertices, RTP_BUFFER_TYPE_HOST, &mesh.positions);*/
			model->setTriangles(
				index_buffer->count(), index_buffer->type(), index_buffer->ptr(),
				vertex_buffer->count(), vertex_buffer->type(), vertex_buffer->ptr(), vertex_buffer->stride()
			);
			model->update(0);
			models.push_back(model); // Model is ref counted, so need to return it to prevent destruction
		}

		prime_instances.reserve(prime_instances.size() + meshes.size());
		for (auto i = 0U; i < meshes.size(); ++i)
		{
			const auto index = model_offset + i;
			auto rtp_model = models[index]->getRTPmodel();
			prime_instances.push_back(rtp_model);
			transforms.emplace_back(meshes[i].transform);
		}
	}

	inline size_t idiv_ceil(const size_t x, const size_t y)
	{
		return (x + y - 1) / y;
	}
}

void bake::ao_optix_prime(
	std::vector<uautil::Mesh> blockers,
	const bake::AOSamples& ao_samples,
	const int rays_per_sample,
	const float scene_offset,
	const float scene_maxdistance,
	const float* bbox_min,
	const float* bbox_max,
	std::vector<float>& ao_values,
	const uautil::Scene& scene)
{
	Timer setup_timer;
	setup_timer.start();

	Context ctx = Context::create(RTP_CONTEXT_TYPE_CUDA);
	std::vector<Model> models;
	std::vector<RTPmodel> prime_instances;
	std::vector<optix::Matrix4x4> transforms;
	std::vector< Buffer<float3>* > allocated_vertex_buffers;
	std::vector< Buffer<int3>* > allocated_index_buffers;
	//if (blockers.num_instances > 0) {
	//	createInstances(ctx, blockers.meshes, blockers.num_meshes, blockers.instances, blockers.num_instances,
	//		allocated_vertex_buffers, allocated_index_buffers, models, prime_instances, transforms);
	//}
	//create_instances(ctx, blockers, true, allocated_vertex_buffers, allocated_index_buffers, models, prime_instances, transforms);
	create_instances(ctx, blockers, true, allocated_vertex_buffers, allocated_index_buffers, models, prime_instances, transforms);
	create_instances(ctx, scene.m_meshes, true, allocated_vertex_buffers, allocated_index_buffers, models, prime_instances, transforms);

	auto scene_model = ctx->createModel();
	scene_model->setInstances(prime_instances.size(), RTP_BUFFER_TYPE_HOST, &prime_instances[0],
		RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4, RTP_BUFFER_TYPE_HOST, &transforms[0]);
	scene_model->update(0);
	auto query = scene_model->createQuery(RTP_QUERY_TYPE_ANY);
	const int sqrt_rays_per_sample = static_cast<int>(sqrtf(static_cast<float>(rays_per_sample)) + .5f);
	setup_timer.stop();
	Timer raygen_timer;
	Timer query_timer;
	Timer updateao_timer;
	Timer copyao_timer;
	//const auto sqrt_rays_per_sample = static_cast<int>(lroundf(sqrtf(static_cast<float>(rays_per_sample))));
	unsigned seed = 0;

	// Split sample points into batches
	const size_t batch_size = 200000; // Note: fits on GTX 750 (1 GB) along with Hunter model
	const auto num_batches = std::max(idiv_ceil(ao_samples.num_samples, batch_size), size_t(1));
	const auto scene_scale = std::max(std::max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]), bbox_max[2] - bbox_min[2]);

	for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++, seed++)
	{
		setup_timer.start();
		const auto sample_offset = batch_idx * batch_size;
		const auto num_samples = std::min(batch_size, ao_samples.num_samples - sample_offset);
		//if (num_samples == 0) continue;

		// Copy all necessary data to device
		Buffer<float3> sample_normals(num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR);
		Buffer<float3> sample_face_normals(num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR);
		Buffer<float3> sample_positions(num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR);
		cudaMemcpy(sample_normals.ptr(), ao_samples.sample_normals + 3 * sample_offset, sample_normals.sizeInBytes(), cudaMemcpyHostToDevice);
		cudaMemcpy(sample_face_normals.ptr(), ao_samples.sample_face_normals + 3 * sample_offset, sample_face_normals.sizeInBytes(), cudaMemcpyHostToDevice);
		cudaMemcpy(sample_positions.ptr(), ao_samples.sample_positions + 3 * sample_offset, sample_positions.sizeInBytes(), cudaMemcpyHostToDevice);
		bake::AOSamples ao_samples_device;
		ao_samples_device.num_samples = num_samples;
		ao_samples_device.sample_normals = reinterpret_cast<float*>(sample_normals.ptr());
		ao_samples_device.sample_face_normals = reinterpret_cast<float*>(sample_face_normals.ptr());
		ao_samples_device.sample_positions = reinterpret_cast<float*>(sample_positions.ptr());
		ao_samples_device.sample_infos = 0;

		Buffer<HitInstancing> hits(num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR);
		Buffer<myRay> rays(num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR);
		Buffer<float> ao(num_samples, RTP_BUFFER_TYPE_CUDA_LINEAR);
		cudaMemset(ao.ptr(), 0, ao.sizeInBytes());

		query->setRays(rays.count(), myRay::format, rays.type(), rays.ptr());
		query->setHits(hits.count(), HitInstancing::format, hits.type(), hits.ptr());
		setup_timer.stop();

		for (int i = 0; i < sqrt_rays_per_sample; ++i) {
			for (int j = 0; j < sqrt_rays_per_sample; ++j)
			{
				ACCUM_TIME(raygen_timer, generate_rays_device(seed, i, j, sqrt_rays_per_sample, scene_offset, scene_maxdistance, ao_samples_device, rays.ptr()));

				// Host or device, depending on Prime context type.  For a host context, which we assume is rare, Prime will copy the rays from device to host.
				ACCUM_TIME(query_timer, query->execute(0));

				ACCUM_TIME(updateao_timer, update_ao_device((int)num_samples, hits.ptr(), ao.ptr(), scene_maxdistance - scene_offset));
			}
		}
		auto distance = hits.hostPtr();
		/*for (size_t i = 0; i < ao_samples.num_samples; i++)
		{
			std::cout << distance[i].t <<"|||" <<distance[i].u<<"|||"<< distance[i].v << std::endl;
		}*/
		copyao_timer.start();
		cudaMemcpy(&ao_values[sample_offset], ao.ptr(), ao.sizeInBytes(), cudaMemcpyDeviceToHost);
		copyao_timer.stop();

	}

	// Normalize
	for (size_t i = 0; i < ao_samples.num_samples; ++i)
	{
		ao_values[i] = 1.0f - ao_values[i] / rays_per_sample;
		//ao_values[i] = ao_values[i] / rays_per_sample;
	}

	// Clean up Buffer pointers. Could be avoided with unique_ptr.
	for (auto& allocated_vertex_buffer : allocated_vertex_buffers)
	{
		delete allocated_vertex_buffer;
	}
	allocated_vertex_buffers.clear();
	for (auto& allocated_index_buffer : allocated_index_buffers)
	{
		delete allocated_index_buffer;
	}
	allocated_index_buffers.clear();

	std::cerr << "\n\tsetup ...           ";  printTimeElapsed(setup_timer);
	std::cerr << "\taccum raygen ...    ";  printTimeElapsed(raygen_timer);
	std::cerr << "\taccum query ...     ";  printTimeElapsed(query_timer);
	std::cerr << "\taccum update AO ... ";  printTimeElapsed(updateao_timer);
	std::cerr << "\tcopy AO out ...     ";  printTimeElapsed(copyao_timer);
}
