#include "kfusion.h"
#include <vector_types.h>
#include "constant_parameters.h"
#include "utils.h"
#include "volume.h"

#include <cstdio>
#include <iostream>
#include <fstream>

#include <cmath>

#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <string.h>

#define KFUSION_INVALID -2

#define KERNEL_TIMINGS
#ifdef KERNEL_TIMINGS
	FILE *kernel_timings_log;
#endif

bool print_kernel_timing = false;

using namespace std; 

#define TICK()    {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}}

#ifndef KERNEL_TIMINGS
#define TOCK(str,size)  {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tock_clockData); std::cerr<< str << " ";\
	if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
	std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}
#else
#define TOCK(str,size) { if(print_kernel_timing) { clock_gettime(CLOCK_MONOTONIC, &tock_clockData);\
							  fprintf(kernel_timings_log,"%s\t%d\t%f\t%f\n",str,size,(double) tick_clockData.tv_sec + tick_clockData.tv_nsec / 1000000000.0,(double) tock_clockData.tv_sec + tock_clockData.tv_nsec / 1000000000.0); }}
#endif


void checkMemAlloc (void * ptr) 
{
	if (ptr == NULL) {
		std::cerr << "Error: Memory allocation problem" << std::endl;
		exit(1);
	}
}

KFusion::KFusion(Configuration config)
	:_frame(-1),
	_tracked(false),
	lastFrame(0),
	lastKeyFrame(0)
{
	if (getenv("KERNEL_TIMINGS"))
		print_kernel_timing = true;
	#ifdef KERNEL_TIMINGS
		print_kernel_timing = true;
		kernel_timings_log = fopen("kernel_timings.log","w");
	#endif

	// kparams_t params;
	params.compute_size_ratio = config.compute_size_ratio;
	params.integration_rate = config.integration_rate;
	params.rendering_rate = config.rendering_rate;
	params.tracking_rate = config.tracking_rate;
	params.volume_resolution = config.volume_resolution;
	params.volume_direction = config.initial_pos;
	params.volume_size = config.volume_size;
	params.mu = config.mu;
	params.icp_threshold = config.icp_threshold;
	params.computationSize = make_uint2(params.inputSize.x/params.compute_size_ratio, params.inputSize.y/params.compute_size_ratio);
	params.camera = config.camera / params.compute_size_ratio;
	params.pyramid = config.pyramid;
	params.keyframe_rate = config.kf_rate;
	params.max_num_kfs = config.max_num_kfs;

	uint3 vr = make_uint3(params.volume_resolution.x,
						  params.volume_resolution.y,
						  params.volume_resolution.z);

	float3 vd = make_float3(params.volume_size.x,
							params.volume_size.y,
							params.volume_size.z);

	volume.init(vr,vd);
	keyFrameVol.init(vr,vd);
	fusionVol.init(vr,vd);

	sMatrix4 initPose;  
	initPose(0,3) = config.initial_pos.x;
	initPose(1,3) = config.initial_pos.y;
	initPose(2,3) = config.initial_pos.z;

	pose = initPose;
	oldPose = pose;

	this->iterations.clear();
	for(auto it = params.pyramid.begin();it != params.pyramid.end(); it++)
	{    
		this->iterations.push_back(*it);
	}

	largestep = 0.75*params.mu;
	inverseCam = getInverseCameraMatrix(params.camera);
	camMatrix = getCameraMatrix(params.camera);
	step = min(params.volume_size) / max(params.volume_resolution);
	viewPose = &pose;

	uint2 cs = make_uint2(params.computationSize.x, params.computationSize.y);	// computation size
	
	std::cout << "Input Size: " << params.inputSize.x << "," << params.inputSize.y << std::endl;
	std::cout << "Computation Size: " << cs.x << "," << cs.y << std::endl;

	// std::cout << "Camera" << "\n" << camMatrix << std::endl;

	reduction = (TrackData*)calloc(cs.x*cs.y*sizeof(TrackData), 1);
	checkMemAlloc(reduction);
	vertex = (float3*)calloc(cs.x*cs.y*sizeof(float3), 1);
	checkMemAlloc(vertex);
	normal = (float3*)calloc(cs.x*cs.y*sizeof(float3), 1);
	checkMemAlloc(normal);
	rawDepth = (float*)calloc(cs.x*cs.y*sizeof(float), 1);
	checkMemAlloc(rawDepth);

	rawRgb = (uchar3*)calloc(params.inputSize.x*params.inputSize.y*sizeof(uchar3), 1);
	checkMemAlloc(rawRgb);

	scaledDepth = (float**) calloc(sizeof(float*) * iterations.size(), 1);
	checkMemAlloc(scaledDepth);
	inputVertex = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);
	checkMemAlloc(inputVertex);
	inputNormal = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);
	checkMemAlloc(inputNormal);

	uint lsize = cs.x * cs.y;
	for (unsigned int i = 0; i < iterations.size(); ++i) {

		scaledDepth[i] = (float*) calloc(sizeof(float) * lsize, 1);
		checkMemAlloc(scaledDepth[i]);

		inputVertex[i] = (float3*) calloc(sizeof(float3) * lsize, 1);
		checkMemAlloc(inputVertex[i]);

		inputNormal[i] = (float3*) calloc(sizeof(float3) * lsize, 1);
		checkMemAlloc(inputNormal[i]);

		lsize = lsize >> 2;
	}

	output = (float*)calloc(32*8*sizeof(float), 1);
	checkMemAlloc(output);

	size_t gaussianS = radius * 2 + 1;
	gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
	checkMemAlloc(gaussian);

	int x;
	for (unsigned int i = 0; i < gaussianS; i++) {
		x = i - radius;
		gaussian[i] = expf(-(x * x) / (2 * delta * delta));
	}

	initVolumeKernel(volume);
	initVolumeKernel(fusionVol);
	initVolumeKernel(keyFrameVol);
	
}

KFusion::~KFusion()
{
	#ifdef KERNEL_TIMINGS
		fclose(kernel_timings_log);
	#endif

	volume.release();
	fusionVol.release();
	keyFrameVol.release();
	
	free(reduction);
	free(normal);
	free(vertex);
	
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		free(scaledDepth[i]);
		free(inputVertex[i]);
		free(inputNormal[i]);
	}
	free(scaledDepth);
	free(inputVertex);
	free(inputNormal);
	
	free(rawDepth);
	free(rawRgb);
	free(output);
	free(gaussian);
	
}

void KFusion::dropKeyFrame(int val)
{
	for(auto it = volumes.begin(); it != volumes.end(); it++) 
	{
		std::cout << "it = " << it->frame << std::endl;
		if(it->frame == val)
		{
			short2 *data=it->data;
			volumes.erase(it); 
			#ifndef CHANGE_VOLS
				delete[] data;
			#endif
			return;
		}
	}
}

void initVolumeKernel(Volume volume) {
	TICK();

	// #pragma omp parallel for schedule(dynamic) 
	for (unsigned int x = 0; x < volume._resolution.x; x++)
		for (unsigned int y = 0; y < volume._resolution.y; y++) {
			for (unsigned int z = 0; z < volume._resolution.z; z++) {
				volume.setints(x, y, z, make_float2(1.0f, 0.0f));
			}
		}
	TOCK("initVolumeKernel", volume._resolution.x*volume._resolution.y*volume._resolution.z);
}


void bilateralFilterKernel(float* out, const float* in, uint2 size,
		const float * gaussian, float e_d, int r) {
	TICK()
	uint y;
	float e_d_squared_2 = e_d * e_d * 2;
	
	#pragma omp parallel for schedule(dynamic) \
			shared(out),private(y)
	for (y = 0; y < size.y; y++) {
		for (uint x = 0; x < size.x; x++) {
			uint pos = x + y * size.x;
			if (in[pos] == 0) {
				out[pos] = 0;
				continue;
			}

			float sum = 0.0f;
			float t = 0.0f;

			const float center = in[pos];

			for (int i = -r; i <= r; ++i) {
				for (int j = -r; j <= r; ++j) {
					uint2 curPos = make_uint2(clamp(x + i, 0u, size.x - 1),
							clamp(y + j, 0u, size.y - 1));
					const float curPix = in[curPos.x + curPos.y * size.x];
					if (curPix > 0) {
						const float mod = (curPix - center)*(curPix - center);
						const float factor = gaussian[i + r]
								* gaussian[j + r]
								* expf(-mod / e_d_squared_2);
						t += factor * curPix;
						sum += factor;
					}
				}
			}
			out[pos] = t / sum;
		}
	}

	TOCK("bilateralFilterKernel", size.x*size.y);
}

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
		const float e_d, const int r) {
	TICK();
	uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
	unsigned int y;

	#pragma omp parallel for schedule(dynamic) \
			shared(out), private(y)
	for (y = 0; y < outSize.y; y++) {
		for (unsigned int x = 0; x < outSize.x; x++) {
			uint2 pixel = make_uint2(x, y);
			const uint2 centerPixel = 2 * pixel;

			float sum = 0.0f;
			float t = 0.0f;
			float center = in[centerPixel.x
					+ centerPixel.y * inSize.x];

			for (int i = -r + 1; i <= r; ++i) {
				for (int j = -r + 1; j <= r; ++j) {
					uint2 cur = make_uint2(
							clamp(
									make_int2(centerPixel.x + j,
											centerPixel.y + i), make_int2(0),
									make_int2(2 * outSize.x - 1,
											2 * outSize.y - 1)));
					float current = in[cur.x + cur.y * inSize.x];
					if (fabsf(current - center) < e_d) {
						sum += 1.0f;
						t += current;
					}
				}
			}
			out[pixel.x + pixel.y * outSize.x] = t / sum;
		}
	}
	TOCK("halfSampleRobustImageKernel", outSize.x*outSize.y);
}

void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize,
		const sMatrix4 invK) {
	TICK();
	unsigned int x, y;
	
	#pragma omp parallel for schedule(dynamic) \
			shared(vertex), private(x, y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {

			if (depth[x + y * imageSize.x] > 0) {
				vertex[x + y * imageSize.x] = depth[x + y * imageSize.x]
						* (rotate(invK, make_float3(x, y, 1.f)));
			} else {
				vertex[x + y * imageSize.x] = make_float3(0);
			}
		}
	}
	TOCK("depth2vertexKernel", imageSize.x*imageSize.y);
}

void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize) {
	TICK();
	unsigned int x, y;
	
	#pragma omp parallel for schedule(dynamic) \
			shared(out), private(x,y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {
			const uint2 pleft = make_uint2(max(int(x) - 1, 0), y);
			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1),
					y);
			const uint2 pup = make_uint2(x, max(int(y) - 1, 0));
			const uint2 pdown = make_uint2(x,
					min(y + 1, ((int) imageSize.y) - 1));

			const float3 left = in[pleft.x + imageSize.x * pleft.y];
			const float3 right = in[pright.x + imageSize.x * pright.y];
			const float3 up = in[pup.x + imageSize.x * pup.y];
			const float3 down = in[pdown.x + imageSize.x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y * imageSize.x].x = KFUSION_INVALID;
				continue;
			}

			const float3 dxv = right - left;
			const float3 dyv = down - up;
			out[x + y * imageSize.x] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
		}
	}
	TOCK("vertex2normalKernel", imageSize.x*imageSize.y);
}


void reduceKernel(float * out, TrackData* J, const uint2 Jsize,
		const uint2 size) {
	TICK();
	int blockIndex;
	
	#pragma omp parallel for schedule(dynamic) private (blockIndex)
	for (blockIndex = 0; blockIndex < 8; blockIndex++) {

		float S[112][32]; // this is for the final accumulation
		// we have 112 threads in a blockdim
		// and 8 blocks in a gridDim?
		// ie it was launched as <<<8,112>>>
		
		for(int threadIndex = 0; threadIndex < 112; threadIndex++) {
			uint sline = threadIndex;
			float sums[32];
			float * jtj = sums+7;
			float * info = sums+28;
			for(uint i = 0; i < 32; ++i) sums[i] = 0;

			for(uint y = blockIndex; y < size.y; y += 8) {
				for(uint x = sline; x < size.x; x += 112) {
					const TrackData & row = J[(x + y * Jsize.x)];

					if (row.result < 1) {
						info[1] += row.result == -4 ? 1 : 0;
						info[2] += row.result == -5 ? 1 : 0;
						info[3] += row.result > -4 ? 1 : 0;
						continue;
					}
					// Error part
					sums[0] += row.error * row.error;

					for(int i = 0; i < 6; ++i)
					sums[i+1] += row.error * row.J[i];

					// JTJ part, unfortunatly the double loop is not unrolled well...
					jtj[0] += row.J[0] * row.J[0];
					jtj[1] += row.J[0] * row.J[1];
					jtj[2] += row.J[0] * row.J[2];
					jtj[3] += row.J[0] * row.J[3];

					jtj[4] += row.J[0] * row.J[4];
					jtj[5] += row.J[0] * row.J[5];

					jtj[6] += row.J[1] * row.J[1];
					jtj[7] += row.J[1] * row.J[2];
					jtj[8] += row.J[1] * row.J[3];
					jtj[9] += row.J[1] * row.J[4];

					jtj[10] += row.J[1] * row.J[5];

					jtj[11] += row.J[2] * row.J[2];
					jtj[12] += row.J[2] * row.J[3];
					jtj[13] += row.J[2] * row.J[4];
					jtj[14] += row.J[2] * row.J[5];

					jtj[15] += row.J[3] * row.J[3];
					jtj[16] += row.J[3] * row.J[4];
					jtj[17] += row.J[3] * row.J[5];

					jtj[18] += row.J[4] * row.J[4];
					jtj[19] += row.J[4] * row.J[5];

					jtj[20] += row.J[5] * row.J[5];

					// extra info here
					info[0] += 1;
				}
			}

			for(int i = 0; i < 32; ++i) {
				S[sline][i] = sums[i];
			}

		}

		for(int ssline = 0; ssline < 32; ssline++) { // sum up columns and copy to global memory in the final 32 threads
			for(unsigned i = 1; i < 112; ++i) {
				S[0][ssline] += S[i][ssline];
			}
			out[ssline+blockIndex*32] = S[0][ssline];
		}
	}

	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
	for (int j = 1; j < 8; ++j) {
		values[0] += values[j];
	}
	TOCK("reduceKernel", 512);


}

void trackKernel(TrackData* output, const float3* inVertex,
		const float3* inNormal, uint2 inSize, const float3* refVertex,
		const float3* refNormal, uint2 refSize, const sMatrix4 Ttrack,
		const sMatrix4 view, const float dist_threshold,
		const float normal_threshold) {
	TICK();
	uint2 pixel = make_uint2(0, 0);
	unsigned int pixely, pixelx;
	
	#pragma omp parallel for schedule(dynamic) \
			shared(output), private(pixel,pixelx,pixely)
	for (pixely = 0; pixely < inSize.y; pixely++) {
		for (pixelx = 0; pixelx < inSize.x; pixelx++) {
			pixel.x = pixelx;
			pixel.y = pixely;

			TrackData & row = output[pixel.x + pixel.y * refSize.x];

			if (inNormal[pixel.x + pixel.y * inSize.x].x == KFUSION_INVALID) {
				row.result = -1;
				continue;
			}

			const float3 projectedVertex = Ttrack
					* inVertex[pixel.x + pixel.y * inSize.x];
			const float3 projectedPos = view * projectedVertex;
			const float2 projPixel = make_float2(
					projectedPos.x / projectedPos.z + 0.5f,
					projectedPos.y / projectedPos.z + 0.5f);
			if (projPixel.x < 0 || projPixel.x > refSize.x - 1
					|| projPixel.y < 0 || projPixel.y > refSize.y - 1) {
				row.result = -2;
				continue;
			}

			const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
			const float3 referenceNormal = refNormal[refPixel.x
					+ refPixel.y * refSize.x];

			if (referenceNormal.x == KFUSION_INVALID) {
				row.result = -3;
				continue;
			}

			const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x]
					- projectedVertex;
			const float3 projectedNormal = rotate(Ttrack,
					inNormal[pixel.x + pixel.y * inSize.x]);

			if (length(diff) > dist_threshold) {
				row.result = -4;
				continue;
			}
			if (dot(projectedNormal, referenceNormal) < normal_threshold) {
				row.result = -5;
				continue;
			}
			row.result = 1;
			row.error = dot(referenceNormal, diff);
			((float3 *) row.J)[0] = referenceNormal;
			((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
		}
	}
	TOCK("trackKernel", inSize.x*inSize.y);
}

void fuseVolumesKernel( Volume dstVol, 
						Volume srcVol, 
						const sMatrix4 pose,
						const float3 origin,
						const float maxweight)
{
	TICK();

	unsigned int y;
	
	float3 vsize = srcVol.getSizeInMeters();

	#pragma omp parallel for schedule(dynamic) \
			shared(dstVol), private(y)
	for (y = 0; y < dstVol.getResolution().y; y++)
		for (unsigned int x = 0; x < dstVol.getResolution().x; x++) {
			uint3 pix = make_uint3(x, y, 0);
			if( pix.x >= dstVol.getResolution().x ||
				pix.y >= dstVol.getResolution().y ) {
				continue;
			}

			for (pix.z = 0; pix.z < dstVol.getResolution().z; pix.z++) {
				float3 pos=dstVol.pos(pix);
				
				pos=pose*pos;
				
				pos.x+=origin.x;
				pos.y+=origin.y;
				pos.z+=origin.z;
				
				if( pos.x<0 || pos.x >= vsize.x ||
					pos.y<0 || pos.y >= vsize.y ||
					pos.z<0 || pos.z >= vsize.z)
				{
					 continue;
				}
				
				float tsdf=srcVol.interp(pos);
				 
				float w_interp=1;
				
				float2 p_data = dstVol[pix];
				
				if(tsdf == 1.0)
					continue;
				
				float w=p_data.y;
				float new_w=w+w_interp;
				
				p_data.x = clamp( (w*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
				p_data.y = fminf(new_w, maxweight);                
				
				dstVol.set(pix,p_data);
			}
		}
	TOCK("fuseVolumesKernel", dstVol.getResolution().y*dstVol.getResolution().x);
}

void integrateKernel(Volume vol, const float* depth, const uchar3* rgb, uint2 depthSize,
		const sMatrix4 invTrack, const sMatrix4 K, const float mu,
		const float maxweight, const int ratio) {
	TICK();
	const float3 delta = rotate(invTrack,
			make_float3(0, 0, vol.dim.z / vol._resolution.z));
	const float3 cameraDelta = rotate(K, delta);
	unsigned int y;

	#pragma omp parallel for schedule(dynamic) \
			shared(vol), private(y)
	for (y = 0; y < vol._resolution.y; y++)
		for (unsigned int x = 0; x < vol._resolution.x; x++) {

			uint3 pix = make_uint3(x, y, 0);
			float3 pos = invTrack * vol.pos(pix);
			float3 cameraX = K * pos;

			for (pix.z = 0; pix.z < vol._resolution.z;
					++pix.z, pos += delta, cameraX += cameraDelta) {

				if (pos.z < 0.0001f) // some near plane constraint
					continue;

				const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
						cameraX.y / cameraX.z + 0.5f);

				if (pixel.x < 0 || pixel.x > depthSize.x - 1 || pixel.y < 0
						|| pixel.y > depthSize.y - 1)
					continue;

				uint2 px = make_uint2(pixel.x, pixel.y);
				if (depth[px.x + px.y * depthSize.x] == 0)
					continue;

				const float diff =
						(depth[px.x + px.y * depthSize.x] - cameraX.z)
								* std::sqrt(
										1 + (pos.x / pos.z)*(pos.x / pos.z)
												+ (pos.y / pos.z)*(pos.y / pos.z));
				if (diff > -mu) {
					const float sdf = fminf(1.f, diff / mu);
					float2 data = vol[pix];

					float w = data.y;
					float new_w = w+1;

					data.x = clamp((w*data.x + sdf)/new_w, -1.f, 1.f);

					data.y = fminf(new_w, maxweight);
					vol.set(pix, data);
				}
			}
		}

	TOCK("integrateKernel", vol._resolution.x*vol._resolution.y*vol._resolution.z);
}

float4 raycast( const Volume volume,
				const uint2 pos,
				const sMatrix4 view,
				const float nearPlane,
				const float farPlane,
				const float step,
				const float largestep)
{
	const float3 origin = view.get_translation();
	const float3 direction = rotate(view, make_float3(pos.x, pos.y, 1.f));

	const float3 invR = make_float3(1.0f) / direction;
	const float3 tbot = -1 * invR * origin;
	const float3 ttop = invR * (volume.getDimensions() - origin);

	// re-order intersections to find smallest and largest on each axis
	const float3 tmin = fminf(ttop, tbot);
	const float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y),
									 fmaxf(tmin.x, tmin.z));
	const float smallest_tmax = fminf(fminf(tmax.x, tmax.y),
									  fminf(tmax.x, tmax.z));

	// check against near and far plane
	const float tnear = fmaxf(largest_tmin, nearPlane);
	const float tfar = fminf(smallest_tmax, farPlane);

	if (tnear < tfar)
	{
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
		float f_t = volume.interp(origin + direction * t);
		float f_tt = 0;

		if (f_t > 0) // ups, if we were already in it, then don't render anything here
		{
			for (; t < tfar; t += stepsize)
			{
				f_tt = volume.interp(origin + direction * t);
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0)
			{           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return make_float4(origin + direction * t, t);
			}
		}
	}
	return make_float4(0);
}

void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
		const Volume integration, const sMatrix4 view, const float nearPlane,
		const float farPlane, const float step, const float largestep) {
	TICK();
	unsigned int y;
#pragma omp parallel for schedule(dynamic) \
		shared(normal, vertex), private(y)
	for (y = 0; y < inputSize.y; y++)
		for (unsigned int x = 0; x < inputSize.x; x++) {

			uint2 pos = make_uint2(x, y);

			const float4 hit = raycast(integration, pos, view, nearPlane,
					farPlane, step, largestep);
			if (hit.w > 0.0) {
				vertex[pos.x + pos.y * inputSize.x] = make_float3(hit);
				float3 surfNorm = integration.grad(make_float3(hit));
				if (length(surfNorm) == 0) {
					normal[pos.x + pos.y * inputSize.x].x = KFUSION_INVALID;
				} else {
					normal[pos.x + pos.y * inputSize.x] = normalize(surfNorm);
				}
			} else {
				vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
				normal[pos.x + pos.y * inputSize.x] = make_float3(KFUSION_INVALID, 0,
						0);
			}
		}
	TOCK("raycastKernel", inputSize.x*inputSize.y);
}


bool KFusion::preprocessing(float *inputDepth, const uchar3 *inputRgb)
{
	// downscale resolution
	int ratio = params.compute_size_ratio;

	if (ratio <= 0) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	if (ratio != 1) {
		unsigned int y;
		#pragma omp parallel for schedule(dynamic) private(y)
		for (y = 0; y < params.computationSize.y; y++)
			for (unsigned int x = 0; x < params.computationSize.x; x++) {
				rawDepth[x + params.computationSize.x * y] = inputDepth[x * ratio + params.inputSize.x * y * ratio];
			}
	}
	else
		memcpy(rawDepth, inputDepth, params.computationSize.x*params.computationSize.y*sizeof(float));

	bilateralFilterKernel(scaledDepth[0], rawDepth, params.computationSize, gaussian, e_delta, radius);
	
	memcpy(rawRgb, inputRgb, params.inputSize.x*params.inputSize.y*sizeof(uchar3));


	return true;
}


bool KFusion::tracking(uint frame)
{
	(void)frame;
	forcePose=false;
	
	// half sample the input depth maps into the pyramid levels
	uint2 localimagesize = params.computationSize;
	for (int i = 1; i < int(iterations.size()); ++i) {
		halfSampleRobustImageKernel(scaledDepth[i], scaledDepth[i-1], 
									localimagesize, e_delta * 3, 1);
		localimagesize = make_uint2(localimagesize.x >> 1, localimagesize.y >> 1);
	}


	float4 k = make_float4(params.camera.x, params.camera.y, params.camera.z, params.camera.w);
	// prepare the 3D information from the input depth maps
	localimagesize = params.computationSize;
	for (int i = 0; i < int(iterations.size()); ++i) {
		
		depth2vertexKernel(inputVertex[i], scaledDepth[i], localimagesize, 
								getInverseCameraMatrix(k / float(1 << i))); // inverse camera matrix depends on level
		vertex2normalKernel( inputNormal[i], inputVertex[i], localimagesize);

		localimagesize = make_uint2(localimagesize.x >> 1, localimagesize.y >> 1);
	}

	oldPose = pose;
	const sMatrix4 projectReference = camMatrix*inverse(sMatrix4(&raycastPose));
	int i;
	for (int level = iterations.size() - 1; level >= 0; --level) {
		uint2 localimagesize = make_uint2(
				params.computationSize.x >> level,
				params.computationSize.y >> level);
		
		for (i = 0; i < iterations[level]; ++i) {
			trackPose=pose;
			trackKernel(   reduction,
						   inputVertex[level],
						   inputNormal[level], localimagesize, 
						   vertex,
						   normal, params.computationSize, 
						   pose,
						   projectReference,
						   dist_threshold,
						   normal_threshold);
			

			reduceKernel(output, reduction, params.computationSize, localimagesize); // compute the linear system to solve
			
			if (updatePoseKernel(pose, output, params.icp_threshold, this->deltaPose))
				break;
		}

	}

	return checkPoseKernel(pose, oldPose, output, params.computationSize, track_threshold);
}

bool KFusion::initKeyFrame(uint frame)
{
	if(frame > 0) {
		VolumeCpu v;
		v.frame = lastKeyFrameIdx;
		v.pose = lastKeyFramePose;
		v.resolution = keyFrameVol.getResolution();
		v.dimensions = keyFrameVol.getDimensions();
		
		uint size = v.resolution.x*v.resolution.y*v.resolution.z;

		v.data = new short2[size];
		
		if(v.data == nullptr) {
			std::cerr << "Error allocating memory." << std::endl;
			exit(1);
		}

		memcpy(v.data, keyFrameVol.getDataPtr(), size*sizeof(short2));
		
		
		volumes.push_back(v);
	}
	
	lastKeyFrameIdx = frame;
	lastKeyFramePose = getPose();
	
	initVolumeKernel(keyFrameVol);
	lastKeyFrame = frame;
	return true;
}

bool KFusion::fuseVolumes()
{        
	//clear volume first    
	initVolumeKernel(volume);
	uint size = fusionVol._resolution.x*fusionVol._resolution.y*fusionVol._resolution.z;
	
	for(int i=0; i < int(volumes.size()); i++) { 
		VolumeCpu &v=volumes[i];  

		memcpy(fusionVol.data, volumes[i].data, size*sizeof(short2));

		fuseVolumesKernel(  volume,
							fusionVol,
							inverse(v.pose),
							params.volume_direction,
							maxweight);
	}
	
	
	#ifdef G2O_OPT_MAX_NUM_KFS
	for (int f_id = 0; f_id < lastKeyFrame; f_id += params.keyframe_rate) {
		dropKeyFrame(f_id);
	}
	#endif

	return true;
}

bool KFusion::fuseLastKeyFrame(sMatrix4 &pose)
{
	lastKeyFramePose = pose;
	fuseVolumesKernel(  volume,
						keyFrameVol,
						inverse(lastKeyFramePose),
						params.volume_direction,
						maxweight);

	return true;
}

bool KFusion::raycasting(uint frame)
{
	if (frame > 2) {
		raycastPose = pose;

		raycastKernel(vertex, normal, params.computationSize, volume, 
						raycastPose * inverseCam,
						nearPlane,
						farPlane,
						step,
						largestep);
	}
	else {
		return false;
	}

	return true;
}

void KFusion::integrateKeyFrameData() {
	sMatrix4 delta = inverse(lastKeyFramePose)*pose;
		
	delta(0,3) += params.volume_direction.x;
	delta(1,3) += params.volume_direction.y;
	delta(2,3) += params.volume_direction.z;
	
	integrateKernel(keyFrameVol, rawDepth, rawRgb, params.computationSize, 
										 inverse(delta), camMatrix, params.mu, maxweight, params.compute_size_ratio);

}

bool KFusion::integration(uint frame)
{
	if (_tracked || _frame <= 3) {
		
		integrateKernel(volume,
						rawDepth,
						rawRgb,
						params.computationSize,
						inverse(pose),
						camMatrix,
						params.mu,
						maxweight, 
						params.compute_size_ratio);
		return true;
	}

	return false;
}

bool KFusion::updatePoseKernel(sMatrix4 & pose, const float * output,float icp_threshold, sMatrix4 &deltaPose)
{
	// Update the pose regarding the tracking result
	TICK();
	bool res = false;

	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	sMatrix4 deltaMat = tosMatrix4(delta);
	sMatrix4 delta4 = deltaMat * sMatrix4(&pose);

	pose.data[0].x = delta4.data[0].x;
	pose.data[0].y = delta4.data[0].y;
	pose.data[0].z = delta4.data[0].z;
	pose.data[0].w = delta4.data[0].w;
	pose.data[1].x = delta4.data[1].x;
	pose.data[1].y = delta4.data[1].y;
	pose.data[1].z = delta4.data[1].z;
	pose.data[1].w = delta4.data[1].w;
	pose.data[2].x = delta4.data[2].x;
	pose.data[2].y = delta4.data[2].y;
	pose.data[2].z = delta4.data[2].z;
	pose.data[2].w = delta4.data[2].w;
	pose.data[3].x = delta4.data[3].x;
	pose.data[3].y = delta4.data[3].y;
	pose.data[3].z = delta4.data[3].z;
	pose.data[3].w = delta4.data[3].w;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold) {
		deltaPose=deltaMat;
		res = true;
	}
	
	TOCK("updatePoseKernel", 1);

	return res;
}

bool KFusion::checkPoseKernel(sMatrix4 & pose,
					 sMatrix4 oldPose,
					 const float * output,
					 uint2 imageSize,
					 float track_threshold)
{
	if(forcePose) {
		_tracked = true;
		return true;
	}
	
	// Check the tracking result, and go back to the previous camera position if necessary
	// return true;
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ( (std::sqrt(values(0, 0) / values(0, 28)) > 2e-2) ||
		 (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold) ) {
		pose = oldPose;
		_tracked = false;
		return false;
	}

	_tracked = true;
	return true;
}


