#ifndef KPARAMS_H
#define KPARAMS_H

#include <vector_types.h>
#include "cutil_math.h"
#include <vector>

typedef struct
{
	int compute_size_ratio=1;
	int integration_rate=1;
	int rendering_rate = 1;
	int tracking_rate=1;

	float optim_thr=1e7;
	float cov_small=1e-4;
	float cov_big=1e-2;

	uint3 volume_resolution = make_uint3(256,256,256);
	float3 volume_direction = make_float3(4.0,4.0,4.0);
	float3 volume_size = make_float3(8,8,8);

	//depth sensors noise covariance
	float cov_z = 0.02;


	std::vector<int> pyramid = {10,5,4};
	float mu = 0.1;
	float icp_threshold = 5.0e-01;

	uint2 inputSize = make_uint2(640, 480);
	uint2 computationSize = make_uint2(640, 480);
	float4 camera = make_float4(570.3422241210938, 319.5, 570.3422241210938, 239.5);

	float rfitness = 0.6;
	float rerror = 1e5;
	int keyframe_rate = 50;
	int max_num_kfs = 5;

} kparams_t;


#endif // KPARAMS_H
