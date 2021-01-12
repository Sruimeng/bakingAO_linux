#pragma once

#include "bake_api.h"

namespace bake{
	int modelToView(
			const std::string out_file,
		const uautil::Scene* scene,
		const size_t num_meshes,
		float** vertex_colors,
		float scene_bbox_min[3],
		float scene_bbox_max[3], std::vector<uautil::Mesh> blockers);
}