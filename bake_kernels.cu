
#include "optix/optixu/optixu_math_namespace.h"
#include "bake_kernels.h"
#include "bake_api.h"
#include "include/cuda/random.h"


using optix::float3;

inline int idiv_ceil(const int x, const int y)
{
	return (x + y - 1) / y;
}

// Ray generation kernel
__global__ void generate_rays_kernel(
	const unsigned int base_seed,
	const int px,
	const int py,
	const int sqrt_passes,
	const float scene_offset,
	const float scene_maxdistance,
	const int num_samples,
	const float3* sample_normals,
	const float3* sample_face_normals,
	const float3* sample_positions,
	myRay* rays)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= num_samples)
		return;

	const unsigned int tea_seed = (base_seed << 16) | (px * sqrt_passes + py);
	unsigned seed = tea<2>(tea_seed, idx);
	//unsigned seed = 1;
	const float3 sample_norm = sample_normals[idx];
	const float3 sample_face_norm = sample_face_normals[idx];
	const float3 sample_pos = sample_positions[idx];
	const float3 ray_origin = sample_pos+scene_offset * sample_norm;
	//const float3 ray_origin = sample_pos ;
	optix::Onb onb(sample_norm);

	float3 ray_dir;
	//float u0 = rnd(seed);
	//float u1 = rnd(seed);
	float u0 = (static_cast<float>(px) + rnd(seed)) / static_cast<float>(sqrt_passes);
	float u1 = (static_cast<float>(py) + rnd(seed)) / static_cast<float>(sqrt_passes);
	int j = 0;
	do
	{
		optix::cosine_sample_hemisphere(u0, u1, ray_dir);

		onb.inverse_transform(ray_dir);
		++j;
		u0 = rnd(seed);
		u1 = rnd(seed);
		//u0 = (static_cast<float>(px) + rnd(seed)) / static_cast<float>(sqrt_passes);
		//u1 = (static_cast<float>(py) + rnd(seed)) / static_cast<float>(sqrt_passes);
	} while (j < 64 && optix::dot(ray_dir, sample_face_norm) <= 0.0f);
	// rays[idx].origin = ray_origin;
	// rays[idx].tmin = 0.0f;
	// rays[idx].direction = -ray_dir;
	//rays[idx].tmax = scene_maxdistance - scene_offset;
#if 1
	// Reverse shadow rays for better performance
	rays[idx].origin = ray_origin;
	rays[idx].tmin = 0.0f;
	rays[idx].direction = ray_dir;
	rays[idx].tmax = scene_maxdistance ;
	//rays[idx].tmax = scene_maxdistance*scene_offset;  // possible loss of precision here (bignum - smallnum)

#else
	// Forward shadow rays for better precision
	rays[idx].origin = -ray_origin;
	rays[idx].tmin = scene_offset;
	rays[idx].direction = -ray_dir;
	rays[idx].tmax = scene_maxdistance ;
#endif
}

__host__ void bake::generate_rays_device(unsigned int seed, int px, int py, int sqrt_passes, float scene_offset, float scene_maxdistance, const bake::AOSamples& ao_samples, myRay* rays)
{
	const int block_size = 512;
	const int block_count = idiv_ceil((int)ao_samples.num_samples, block_size);

	generate_rays_kernel << <block_count, block_size >> > (
		seed,
		px,
		py,
		sqrt_passes,
		scene_offset,
		scene_maxdistance,
		(int)ao_samples.num_samples,
		(float3*)ao_samples.sample_normals,
		(float3*)ao_samples.sample_face_normals,
		(float3*)ao_samples.sample_positions,
		rays
		);
}

// AO update kernel
__global__ void update_ao_kernel(int num_samples, HitInstancing* hit_data, float* ao_data, float maxdistance)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx >= num_samples) return;


	float distance = hit_data[idx].t;

	ao_data[idx] += distance > 0.0? 1.0f : 0.0f;
}

// Precondition: ao output initialized to 0 before first pass
__host__ void bake::update_ao_device(int num_samples, HitInstancing* hits, float* ao, float maxdistance)
{
	int block_size = 512;
	int block_count = idiv_ceil(num_samples, block_size);
	update_ao_kernel << <block_count, block_size >> > (num_samples, hits, ao, maxdistance);
}
