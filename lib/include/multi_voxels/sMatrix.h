#ifndef SMATRIX_H 
#define SMATRIX_H

#include <iostream>
#include <ostream>

#include <vector_types.h>
#include "cutil_math.h"

#ifdef INT_HP
typedef struct  {
	floatH4 data[4];
} sMatrix4H;
#endif

struct sMatrix4
{
	float4 data[4];

	//Identity matrix
	sMatrix4()
	{
		this->data[0] = make_float4(1,0,0,0);
		this->data[1] = make_float4(0,1,0,0);
		this->data[2] = make_float4(0,0,1,0);
		this->data[3] = make_float4(0,0,0,1);
	}

	sMatrix4(float x1,float y1,float z1,float w1,
			float x2,float y2,float z2,float w2,
			float x3,float y3,float z3,float w3,
			float x4,float y4,float z4,float w4
	)
	{
		data[0]=make_float4(x1,y1,z1,w1);
		data[1]=make_float4(x2,y2,z2,w2);
		data[2]=make_float4(x3,y3,z3,w3);
		data[3]=make_float4(x4,y4,z4,w4);
	}

	/*This is stupid. Just use two dimensions array*/
	float& operator () (int i,int j)
	{
		if(j==0)
			return this->data[i].x;
		else if(j==1)
			return this->data[i].y;
		else if(j==2)
			return this->data[i].z;
		else //if(j==3)
			return this->data[i].w;
	}


	/*This is also stupid.*/
	const float& operator () (int i,int j) const
	{
		if(j==0)
			return this->data[i].x;
		else if(j==1)
			return this->data[i].y;
		else if(j==2)
			return this->data[i].z;
		else //if(j==3)
			return this->data[i].w;
	}

	sMatrix4(sMatrix4 * src)
	{
		this->data[0] = make_float4(src->data[0].x, src->data[0].y,
				src->data[0].z, src->data[0].w);
		this->data[1] = make_float4(src->data[1].x, src->data[1].y,
				src->data[1].z, src->data[1].w);
		this->data[2] = make_float4(src->data[2].x, src->data[2].y,
				src->data[2].z, src->data[2].w);
		this->data[3] = make_float4(src->data[3].x, src->data[3].y,
				src->data[3].z, src->data[3].w);
	}

	inline  float3 get_translation() const
	{
		return make_float3(data[0].w, data[1].w, data[2].w);
	}

};


struct sMatrix6
{
	float data[6*6];

	sMatrix6(float a00,float a01,float a02,float a03,float a04,float a05,
								 float a10,float a11,float a12,float a13,float a14,float a15,
								 float a20,float a21,float a22,float a23,float a24,float a25,
								 float a30,float a31,float a32,float a33,float a34,float a35,
								 float a40,float a41,float a42,float a43,float a44,float a45,
								 float a50,float a51,float a52,float a53,float a54,float a55)
	{
		data[0]=a00;
		data[1]=a01;
		data[2]=a02;
		data[3]=a03;
		data[4]=a04;
		data[5]=a05;

		data[6] =a10;
		data[7] =a11;
		data[8] =a12;
		data[9] =a13;
		data[10]=a14;
		data[11]=a15;

		data[12]=a20;
		data[13]=a21;
		data[14]=a22;
		data[15]=a23;
		data[16]=a24;
		data[17]=a25;

		data[18]=a30;
		data[19]=a31;
		data[20]=a32;
		data[21]=a33;
		data[22]=a34;
		data[23]=a35;

		data[24]=a40;
		data[25]=a41;
		data[26]=a42;
		data[27]=a43;
		data[28]=a44;
		data[29]=a45;

		data[30]=a50;
		data[31]=a51;
		data[32]=a52;
		data[33]=a53;
		data[34]=a54;
		data[35]=a55;
	}

	sMatrix6()
	{
		for(int i=0;i<6;i++)
		{
			for(int j=0;j<6;j++)
			{
				int idx=6*i+j;
				if(i==j)
					data[idx]=1.0;
				else
					data[idx]=0.0;
			}
		}
	}

	static sMatrix6 zeros()
	{
		sMatrix6 ret;
		for(int i=0;i<6*6;i++)
			ret.data[i]=0.0;
		return ret;
	}

	inline  float& operator () (int i,int j)
	{
		int idx=6*i+j;
		return data[idx];
	}

	inline  const float& operator () (int i,int j) const
	{
		int idx=6*i+j;
		return data[idx];
	}
};

class sMatrix3
{
	public:
	float data[3*3];
	sMatrix3()
	{
		for(int i=0;i<3;i++)
		{
			for(int j=0;j<3;j++)
			{
				int idx=3*i+j;
				if(i==j)
					data[idx]=1.0;
				else
					data[idx]=0.0;
			}
		}
	}

	inline  float& operator () (int i,int j)
	{
		int idx=3*i+j;
		return data[idx];
	}

	inline  const float& operator () (int i,int j) const
	{
		int idx=3*i+j;
		return data[idx];
	}

	static sMatrix3 zeros()
	{
		sMatrix3 mat;
		for(int i=0;i<3*3;i++)
			mat.data[i]=0.0;
		return mat;
	}
};

inline  float4 operator*(const sMatrix4 & M,const float4 & v)
{
	return make_float4(dot(M.data[0], v), dot(M.data[1], v), dot(M.data[2], v),
			dot(M.data[3], v));
}

inline  float3 operator*(const sMatrix4 & M,const float3 & v)
{
	return make_float3(dot(make_float3(M.data[0]), v) + M.data[0].w,
			dot(make_float3(M.data[1]), v) + M.data[1].w,
			dot(make_float3(M.data[2]), v) + M.data[2].w);
}

#ifdef INT_HP
inline floatH3 operator*(const sMatrix4H & M,
		const floatH3 & v) {
	return make_floatH3(dot(make_floatH3(M.data[0]), v) + M.data[0].w,
			dot(make_floatH3(M.data[1]), v) + M.data[1].w,
			dot(make_floatH3(M.data[2]), v) + M.data[2].w);
}
#endif

inline  float3 rotate(const sMatrix4 & M, const float3 & v)
{
	return make_float3(dot(make_float3(M.data[0]), v),
					   dot(make_float3(M.data[1]), v),
					   dot(make_float3(M.data[2]), v));
}

inline float3 rotate(const sMatrix4 & M, const float4 & v4) {
	float3 v = make_float3(v4.x, v4.y, v4.z);
	return make_float3(dot(make_float3(M.data[0]), v),
			dot(make_float3(M.data[1]), v), dot(make_float3(M.data[2]), v));
}

inline  sMatrix6 operator+(const sMatrix6 &c1, const sMatrix6 &c2)
{
	sMatrix6 ret;
	for(int i=0;i<36;i++)
	{
		ret.data[i]=c1.data[i]+c2.data[i];
	}
	return ret;
}

inline  sMatrix4 operator-(const sMatrix4 &c1, const sMatrix4 &c2)
{
	sMatrix4 ret;
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			ret(i,j)=c1(i,j)-c2(i,j);
	return ret;
}

inline  sMatrix3 operator-(const sMatrix3 &c1, const sMatrix3 &c2)
{
	sMatrix3 ret;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			ret(i,j)=c1(i,j)-c2(i,j);
	return ret;
}

#endif
