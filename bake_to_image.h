#pragma once

#include "bake_api.h"
#include "config.h"
namespace bake{
	int modelToView(
		const Config* config,
		uautil::Scene* scene,
		const size_t num_meshes,
		float** vertex_colors,
		float scene_bbox_min[3],
		float scene_bbox_max[3], std::vector<uautil::Mesh> blockers);
}

