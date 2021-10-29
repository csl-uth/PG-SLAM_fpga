#ifndef VOLUME_H
#define VOLUME_H

#include"cutil_math.h"
#include"kparams.h"
#include<iostream>
#include"sMatrix.h"


struct VolumeCpu
{
	uint frame;
	sMatrix4 pose;
	uint3 resolution;
	float3 dimensions;

	short2 *data;
};


struct out_data
{
	char c[6];
};

struct Volume
{
	typedef float (Volume::*Fptr)(const int3&) const;

	Volume()
	{
		_resolution = make_uint3(0);
		dim = make_float3(1);
		data = nullptr;
	}

	bool isNull() const
	{
		return data == nullptr;
	}

	uint3 getResolution() const
	{
		return _resolution;
	}

	float3 getVoxelSize() const
	{
		return voxelSize;
	}

	float3 getSizeInMeters() const
	{
		return make_float3(voxelSize.x * _resolution.x,
						   voxelSize.y * _resolution.y,
						   voxelSize.z * _resolution.z);
	}

	int3 getOffset() const
	{
		return _offset;
	}

	float3 getOffsetPos() const
	{
		return make_float3(_offset.x * voxelSize.x,
						   _offset.y * voxelSize.y,
						   _offset.z * voxelSize.z);
	}

	float3 getDimWithOffset() const
	{
		int3 v = maxVoxel();
		float3 ret;
		ret.x = (v.x) * voxelSize.x;
		ret.y = (v.y) * voxelSize.y;
		ret.z = (v.z) * voxelSize.z;
		return ret;
	}

	float3 center() const
	{
		return make_float3(float(_resolution.x)*voxelSize.x*0.5 + float(_offset.x)*voxelSize.x,
						   float(_resolution.y)*voxelSize.x*0.5 + float(_offset.y)*voxelSize.y,
						   float(_resolution.z)*voxelSize.x*0.5 + float(_offset.z)*voxelSize.z);
	}

	void addOffset(int3 off)
	{
		_offset.x += off.x;
		_offset.y += off.y;
		_offset.z += off.z;
	}

	short2*  getDataPtr() const
	{
		return data;
	}

	float3 getDimensions() const
	{
		return dim;
	}

	uint getIdx(const uint3 &pos) const
	{
		return pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
	}

	uint getIdx(const int3 &pos) const
	{
		return pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
	}

	
	float2 operator[](const int3 & pos) const
	{
		const short2 d = data[getIdx(pos)];
		return make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f
	}

	
	float2 operator[](const uint3 & pos) const
	{
		uint p = pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
		const short2 d = data[p];
		return make_float2(d.x * 0.00003051944088f, d.y); //  / 32766.0f
	}
	
	float vs(const int3 & pos) const
	{
		return data[getIdx(pos)].x;
	}
	
	
	float vw(const int3 & pos) const
	{
		return data[getIdx(pos)].y;
	}
	
	
	float vww(const int3 & pos) const
	{
		short w = data[getIdx(pos)].y;
		if(w > 0)
			return 1.0;
		return 0.0;
	}

	
	void set(const int3 & pos, const float2 & d)
	{
		size_t idx = getIdx(pos);
		data[idx] = make_short2(d.x * 32766.0f, d.y);
	}

	
	void set(const uint3 & pos, const float2 & d)
	{
		uint p = pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
		data[p] = make_short2(d.x * 32766.0f, d.y);
	}

	
	void set(const int3 & pos, const float2 &d,const float3 &c)
	{
		size_t p = getIdx(pos);
		data[p] = make_short2(d.x * 32766.0f, d.y);
	}

	
	void set(const uint3 & pos, const float2 &d,const float3 &c)
	{
		uint p = pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y;
		data[p] = make_short2(d.x * 32766.0f, d.y);
	}

	void setints(const int x, const int y, const int z, const float2 & d)
	{
		int3 pos = make_int3(x, y, z);
		size_t idx = getIdx(pos);
		data[idx] = make_short2(d.x * 32766.0f, d.y);
	}
	
	float3 pos(const int3 & p) const
	{
		return make_float3( ( (p.x + 0.5f) * voxelSize.x),
							( (p.y + 0.5f) * voxelSize.y),
							( (p.z + 0.5f) * voxelSize.z));
	}

	
	float3 pos(const uint3 & p) const
	{
		return make_float3( ( (p.x + 0.5f) * voxelSize.x),
							( (p.y + 0.5f) * voxelSize.y),
							( (p.z + 0.5f) * voxelSize.z));
	}

	
	inline float vs2(const uint x, const uint y, const uint z) const {
		return data[x + y * _resolution.x + z * _resolution.x * _resolution.y].x;
	}

	float interp(const float3 & pos) const;
	
	float red_interp(const float3 & pos) const;

	float green_interp(const float3 & pos) const;

	float blue_interp(const float3 & pos) const;

	float3 grad(const float3 & pos) const;
	
	float3 rgb_interp(const float3 &p) const;

	void init(uint3 resolution, float3 dimensions, short2* ptr);

	int3 minVoxel() const
	{
		return _offset;
	}

	int3 maxVoxel() const
	{
		return make_int3( int(_resolution.x) + _offset.x,
						  int(_resolution.y) + _offset.y,
						  int(_resolution.z) + _offset.z);
	}

	void release()
	{
		if(data != nullptr)
			free(data);

		data = nullptr;
	}

	uint3 _resolution;
	float3 dim;
	float3 voxelSize;
	int3 _offset;

	short2 *data;
};

//Usefull functions
void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData);
void saveVoxelsToFile(char *fileName,const Volume volume);

// #ifdef CHANGE_VOLS
// struct VolumeCpu
// {
// 	uint frame;
// 	sMatrix4 pose;
// 	struct Volume vlm;
// };
// #endif


#endif // VOLUME_H
