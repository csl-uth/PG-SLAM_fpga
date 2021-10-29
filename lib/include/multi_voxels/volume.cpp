#include"volume.h"
#include"marching_cube.h"
#include<iostream>
#include"cutil_math.h"
#include<fstream>
#include "kparams.h"
#include <string>
#include <string.h>

void Volume::init(uint3 resolution, float3 dimensions) {
	_resolution = resolution;
	dim = dimensions;
	
	uint size = _resolution.x * _resolution.y * _resolution.z;
	
	data = (short2*)malloc(size*sizeof(short2));
	if (data == NULL) {
		std::cerr << "Error: Memory allocation problem" << std::endl;
		exit(1);
	}

	voxelSize = dim/_resolution;

	_offset = make_int3(0,0,0);
}

float Volume::interp(const float3 & pos) const {

	const float3 scaled_pos = make_float3((pos.x * _resolution.x / dim.x) - 0.5f,
			(pos.y * _resolution.y / dim.y) - 0.5f,
			(pos.z * _resolution.z / dim.z) - 0.5f);
	const int3 base = make_int3(floorf(scaled_pos));
	const float3 factor = fracf(scaled_pos);
	const int3 lower = max(base, make_int3(0));
	const int3 upper = min(base + make_int3(1),
			make_int3(_resolution) - make_int3(1));
	return (((vs2(lower.x, lower.y, lower.z) * (1 - factor.x)
			+ vs2(upper.x, lower.y, lower.z) * factor.x) * (1 - factor.y)
			+ (vs2(lower.x, upper.y, lower.z) * (1 - factor.x)
					+ vs2(upper.x, upper.y, lower.z) * factor.x) * factor.y)
			* (1 - factor.z)
			+ ((vs2(lower.x, lower.y, upper.z) * (1 - factor.x)
					+ vs2(upper.x, lower.y, upper.z) * factor.x)
					* (1 - factor.y)
					+ (vs2(lower.x, upper.y, upper.z) * (1 - factor.x)
							+ vs2(upper.x, upper.y, upper.z) * factor.x)
							* factor.y) * factor.z) * 0.00003051944088f;

}


float3 Volume::grad(const float3 & pos) const {
	const float3 scaled_pos = make_float3((pos.x * _resolution.x / dim.x) - 0.5f,
										  (pos.y * _resolution.y / dim.y) - 0.5f,
										  (pos.z * _resolution.z / dim.z) - 0.5f);
	const int3 base = make_int3(floorf(scaled_pos));
	const float3 factor = fracf(scaled_pos);

	const int3 lower_lower = max(base - make_int3(1), _offset);
	const int3 lower_upper = max(base, _offset);

	const int3 upper_lower = min(base + make_int3(1),
								 maxVoxel() - make_int3(1));

	const int3 upper_upper = min(base + make_int3(2),
								 maxVoxel() - make_int3(1));

	const int3 & lower = lower_upper;
	const int3 & upper = upper_lower;

	float3 gradient;

	gradient.x = ((
		( vs(make_int3(upper_lower.x, lower.y, lower.z))-vs(make_int3(lower_lower.x, lower.y, lower.z))) * (1 - factor.x)
		+ ( vs(make_int3(upper_upper.x, lower.y, lower.z))-vs(make_int3(lower_upper.x, lower.y, lower.z))) * factor.x) * (1 - factor.y)
		+ ( (vs(make_int3(upper_lower.x, upper.y, lower.z)) - vs(make_int3(lower_lower.x, upper.y, lower.z)))* (1 - factor.x)
			+ (vs(make_int3(upper_upper.x, upper.y, lower.z))- vs(make_int3(lower_upper.x, upper.y,lower.z))) * factor.x) * factor.y) * (1 - factor.z)
				 + (((vs(make_int3(upper_lower.x, lower.y, upper.z))
					  - vs(make_int3(lower_lower.x, lower.y, upper.z)))
					 * (1 - factor.x)
					 + (vs(make_int3(upper_upper.x, lower.y, upper.z))
						- vs(
							make_int3(lower_upper.x, lower.y,
									   upper.z))) * factor.x)
					* (1 - factor.y)
					+ ((vs(make_int3(upper_lower.x, upper.y, upper.z))
						- vs(
							make_int3(lower_lower.x, upper.y,
									   upper.z))) * (1 - factor.x)
					   + (vs(
							  make_int3(upper_upper.x, upper.y,
										 upper.z))
						  - vs(
							  make_int3(lower_upper.x,
										 upper.y, upper.z)))
					   * factor.x) * factor.y) * factor.z;

	gradient.y =
			(((vs(make_int3(lower.x, upper_lower.y, lower.z))
			   - vs(make_int3(lower.x, lower_lower.y, lower.z)))
			  * (1 - factor.x)
			  + (vs(make_int3(upper.x, upper_lower.y, lower.z))
				 - vs(
					 make_int3(upper.x, lower_lower.y,
								lower.z))) * factor.x)
			 * (1 - factor.y)
			 + ((vs(make_int3(lower.x, upper_upper.y, lower.z))
				 - vs(
					 make_int3(lower.x, lower_upper.y,
								lower.z))) * (1 - factor.x)
				+ (vs(
					   make_int3(upper.x, upper_upper.y,
								  lower.z))
				   - vs(
					   make_int3(upper.x,
								  lower_upper.y, lower.z)))
				* factor.x) * factor.y) * (1 - factor.z)
			+ (((vs(make_int3(lower.x, upper_lower.y, upper.z))
				 - vs(
					 make_int3(lower.x, lower_lower.y,
								upper.z))) * (1 - factor.x)
				+ (vs(
					   make_int3(upper.x, upper_lower.y,
								  upper.z))
				   - vs(
					   make_int3(upper.x,
								  lower_lower.y, upper.z)))
				* factor.x) * (1 - factor.y)
			   + ((vs(
					   make_int3(lower.x, upper_upper.y,
								  upper.z))
				   - vs(
					   make_int3(lower.x,
								  lower_upper.y, upper.z)))
				  * (1 - factor.x)
				  + (vs(
						 make_int3(upper.x,
									upper_upper.y, upper.z))
					 - vs(
						 make_int3(upper.x,
									lower_upper.y,
									upper.z)))
				  * factor.x) * factor.y)
			* factor.z;

	gradient.z = (((vs(make_int3(lower.x, lower.y, upper_lower.z))
					- vs(make_int3(lower.x, lower.y, lower_lower.z)))
				   * (1 - factor.x)
				   + (vs(make_int3(upper.x, lower.y, upper_lower.z))
					  - vs(make_int3(upper.x, lower.y, lower_lower.z)))
				   * factor.x) * (1 - factor.y)
				  + ((vs(make_int3(lower.x, upper.y, upper_lower.z))
					  - vs(make_int3(lower.x, upper.y, lower_lower.z)))
					 * (1 - factor.x)
					 + (vs(make_int3(upper.x, upper.y, upper_lower.z))
						- vs(
							make_int3(upper.x, upper.y,
									   lower_lower.z))) * factor.x)
				  * factor.y) * (1 - factor.z)
				 + (((vs(make_int3(lower.x, lower.y, upper_upper.z))
					  - vs(make_int3(lower.x, lower.y, lower_upper.z)))
					 * (1 - factor.x)
					 + (vs(make_int3(upper.x, lower.y, upper_upper.z))
						- vs(
							make_int3(upper.x, lower.y,
									   lower_upper.z))) * factor.x)
					* (1 - factor.y)
					+ ((vs(make_int3(lower.x, upper.y, upper_upper.z))
						- vs(
							make_int3(lower.x, upper.y,
									   lower_upper.z)))
					   * (1 - factor.x)
					   + (vs(
							  make_int3(upper.x, upper.y,
										 upper_upper.z))
						  - vs(
							  make_int3(upper.x, upper.y,
										 lower_upper.z)))
					   * factor.x) * factor.y) * factor.z;

	return gradient
			* make_float3(dim.x / _resolution.x, dim.y / _resolution.y, dim.z / _resolution.z)
			* (0.5f * 0.00003051944088f);
}

void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData)
{
	int3 min=volume.minVoxel();
	int3 max=volume.maxVoxel();
	for(int z=min.z; z<max.z-1; z++)
	{
		for(int y=min.y; y<max.y-1; y++)
		{
			for(int x=min.x;x<max.x-1;x++)
			{
				//Loop over all cubes
				const uint8_t cubeIndex = getCubeIndex(x,y,z,volume, hostData);
				const int* tri = triTable[cubeIndex];

				for(int i=0; i<5; i++)
				{
					if(tri[3*i]<0)
						break;

					float3 p1 = calcPtInterpolate(tri[3*i],x, y, z, volume,hostData);
					float3 p2 = calcPtInterpolate(tri[3*i+1],x, y, z, volume,hostData);
					float3 p3 = calcPtInterpolate(tri[3*i+2],x, y, z, volume,hostData);

					triangles.push_back(p1);
					triangles.push_back(p2);
					triangles.push_back(p3);
				}
			}
		}
	}
}

// void saveVoxelsToFile(char *fileName,const Volume volume,const kparams_t &params)
void saveVoxelsToFile(char *fileName, const Volume volume)
{
	//TODO this function needs cleanup and speedup
	std::cout<<"Saving TSDF voxel grid values to disk("<<fileName<<")"<< std::endl;

	std::ofstream outFile(fileName, std::ios::out);
	float dimensions[3];
	dimensions[0] = float(volume.getResolution().x);
	dimensions[1] = float(volume.getResolution().y);
	dimensions[2] = float(volume.getResolution().z);

	outFile << dimensions[0] << std::endl;
	outFile << dimensions[1] << std::endl;
	outFile << dimensions[2] << std::endl;

	//assuming cubical voxels
	float vox_size = volume.getVoxelSize().x;
	
	outFile << vox_size << std::endl;
	
	uint3 pos;
	for (pos.z = 0; pos.z < volume.getResolution().z; pos.z++)
	{
		for (pos.y = 0; pos.y < volume.getResolution().y; pos.y++)
		{
			for (pos.x = 0; pos.x < volume.getResolution().x; pos.x++)
			{
				uint arrayPos = volume.getIdx(pos);
				short2 data = volume.data[arrayPos];
				float value = float(data.x)/32766.0f;
			}
		}
	}

	outFile.close();

	std::cout<<"Saving done."<<std::endl;
}

