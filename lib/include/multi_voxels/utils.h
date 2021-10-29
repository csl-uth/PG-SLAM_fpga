#ifndef UTILS_H
#define UTILS_H

#include<limits>

#if defined(__GNUC__)
// circumvent packaging problems in gcc 4.7.0
#undef _GLIBCXX_ATOMIC_BUILTINS 
#undef _GLIBCXX_USE_INT128 
// need c headers for __int128 and uint16_t
#include <limits.h>
#endif

#include <stdint.h>
#include <iostream>
#include <vector>

#include <vector_types.h>
#include "cutil_math.h"

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>

#include <iomanip>

#include"sMatrix.h"


#define INVALID -2   // this is used to mark invalid entries in normal or vertex maps

struct TrackData
{
	int result;
	float error;
	float J[6];
};


class Timestamp
{
	public:
		uint32_t sec;
		uint32_t nsec;
		
		Timestamp(uint32_t s,uint32_t ns) :sec(s),nsec(ns){}
};


extern bool print_kernel_timing;

extern struct timespec tick_clockData;
extern struct timespec tock_clockData;


float sq(float r);
double sq(double r);

inline int divup(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline dim3 divup(uint2 a, dim3 b)
{
	return dim3(divup(a.x, b.x), divup(a.y, b.y));
}
inline dim3 divup(dim3 a, dim3 b)
{
	return dim3(divup(a.x, b.x), divup(a.y, b.y), divup(a.z, b.z));
}

extern sMatrix4 T_B_P;
extern sMatrix4 invT_B_P;

inline sMatrix4 getCameraMatrix(const float4 & k) {
	sMatrix4 K;
	K.data[0] = make_float4(k.x, 0, k.z, 0);
	K.data[1] = make_float4(0, k.y, k.w, 0);
	K.data[2] = make_float4(0, 0, 1, 0);
	K.data[3] = make_float4(0, 0, 0, 1);
	return K;
}

inline sMatrix4 getInverseCameraMatrix(const float4 & k)
{
	sMatrix4 invK;
	invK.data[0] = make_float4(1.0f / k.x, 0, -k.z / k.x, 0);
	invK.data[1] = make_float4(0, 1.0f / k.y, -k.w / k.y, 0);
	invK.data[2] = make_float4(0, 0, 1, 0);
	invK.data[3] = make_float4(0, 0, 0, 1);
	return invK;
}

template<typename P>
inline sMatrix4 tosMatrix4(const TooN::SE3<P> & p)
{
	const TooN::Matrix<4, 4, float> I = TooN::Identity;
	sMatrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = p * I;
	return R;
}

template<typename P, typename A>
TooN::Matrix<6> makeJTJ(const TooN::Vector<21, P, A> & v)
{
	TooN::Matrix<6> C = TooN::Zeros;
	C[0] = v.template slice<0, 6>();
	C[1].template slice<1, 5>() = v.template slice<6, 5>();
	C[2].template slice<2, 4>() = v.template slice<11, 4>();
	C[3].template slice<3, 3>() = v.template slice<15, 3>();
	C[4].template slice<4, 2>() = v.template slice<18, 2>();
	C[5][5] = v[20];

	for (int r = 1; r < 6; ++r)
		for (int c = 0; c < r; ++c)
			C[r][c] = C[c][r];

	return C;
}

template<typename T, typename A>
TooN::Vector<6> solve(const TooN::Vector<27, T, A> & vals) {
	const TooN::Vector<6> b = vals.template slice<0, 6>();
	const TooN::Matrix<6> C = makeJTJ(vals.template slice<6, 21>());

	TooN::GR_SVD<6, 6> svd(C);
	return svd.backsub(b, 1e6);
}


inline sMatrix4 operator*(const sMatrix4 & A, const sMatrix4 & B)
{
	sMatrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = TooN::wrapMatrix<4, 4>(&A.data[0].x)
			* TooN::wrapMatrix<4, 4>(&B.data[0].x);
	return R;
}

sMatrix6 operator*(const sMatrix6 & A, const sMatrix6 & B);

sMatrix6 operator*(const sMatrix6 & A, const float f);

sMatrix3 operator*(const sMatrix3 & A, const float f);

sMatrix3 wedge(float *v);

sMatrix3 transpose( const sMatrix3 &mat);

inline std::ostream & operator<<(std::ostream & out, const sMatrix4 & m)
{
	for (unsigned i = 0; i < 4; ++i)
		out << m.data[i].x << "  " << m.data[i].y << "  " << m.data[i].z << "  "
			<< m.data[i].w << "\n";
	return out;
}

inline std::ostream & operator<<(std::ostream & out, const float3 f)
{
	out<<"("<<f.x<<","<<f.y<<","<<f.z<<")";
	return out;
}

inline std::ostream & operator<<(std::ostream & out, const int3 f)
{
	out<<"("<<f.x<<","<<f.y<<","<<f.z<<")";
	return out;
}

inline std::ostream & operator<<(std::ostream & out, const sMatrix6 & m)
{
	for (unsigned i = 0; i < 6; ++i)
	{
		for (unsigned j = 0; j < 6; ++j)
		{
			out << m(i,j)<< "  ";
		}
		out<<"\n";
	}

	return out;
}

inline std::ostream & operator<<(std::ostream & out, const sMatrix3 & m)
{
	for (unsigned i = 0; i < 3; ++i)
	{
		for (unsigned j = 0; j < 3; ++j)
		{
			out << m(i,j)<< "  ";
		}
		out<<"\n";
	}

	return out;
}

inline sMatrix6 tr(const sMatrix6 &mat)
{
	sMatrix6 ret;
	for (unsigned i = 0; i < 6; ++i)
	{
		for (unsigned j = 0; j < 6; ++j)
		{
			ret(j,i)=mat(i,j);
		}
	}
	return ret;
}

inline sMatrix4 fromVisionCord(const sMatrix4 &mat)
{
	return invT_B_P*mat*T_B_P;
//    return ret;
}

inline sMatrix4 toVisionCord(const sMatrix4 &mat)
{
	return T_B_P*mat*invT_B_P;
}

inline float3 fromVisionCordV(const float3 &v)
{
	float3 ret=make_float3( v.z,-v.x,-v.y);
	return ret;
}

inline float3 toVisionCordV(const float3 &v)
{
	sMatrix4 mat;
	mat(0,3)=v.x;
	mat(1,3)=v.y;
	mat(2,3)=v.z;

	mat=toVisionCord(mat);
	float3 ret=mat.get_translation();
	return ret;
}

inline bool isNum(bool b)
{
	if(b!=b) //nan
		return false;
	else if(b==std::numeric_limits<double>::infinity()) //positive inf
		return false;
	else if(b==-std::numeric_limits<double>::infinity()) //positive inf
		return false;
	return true;
}

inline bool isNumf(float b)
{
	if(b!=b) //nan
		return false;
	else if(b==std::numeric_limits<float>::infinity()) //positive inf
		return false;
	else if(b==-std::numeric_limits<float>::infinity()) //positive inf
		return false;
	return true;
}


inline double l2(float *f1,float *f2, int size)
{
	double dist=0;
	double max=std::numeric_limits<double>::max();

	for(int i=0;i<size;i++)
	{
		if(!isNumf(f1[i]))
			std::cout<<"F1 "<<f1[i]<<std::endl;
		if(!isNumf(f2[i]))
			std::cout<<"F2 "<<f2[i]<<std::endl;

		double tmp=sq(f1[i]-f2[i]);
		if(!isNum(tmp))
			std::cout<<"tmp "<<f1[i]<<" "<<f2[i]<<tmp<<std::endl;
		if(max-dist <= tmp )
			return std::numeric_limits<double>::infinity();

		dist+=tmp;
	}

	dist=sqrt(dist);
	return dist;
}

inline float l2(float3 p)
{
	double sum=sq(p.x)+sq(p.y)+sq(p.z);
	
	float ret=(float)sqrt(sum);
	return ret;
}

inline float trace(sMatrix3 m)
{
	float ret=0.0;
	for(int i=0;i<3;i++)
		ret+=m(i,i);

	return ret;
}

inline float3 vec(sMatrix3 M)
{
	float3 v;
	v.x = M(2,1);
	v.y = M(0,2);
	v.z = M(1,0);
	return v;
}


inline float3 logMap(sMatrix3 Rt)
{
	float3 res = make_float3(0,0,0);
	float costheta = (trace(Rt)-1.0)/2.0;
	float theta = acos(costheta);

	if (theta != 0.000)
	{
		sMatrix3 lnR = sMatrix3::zeros();
		lnR =  Rt - transpose(Rt);
		lnR = lnR * ( theta /(2.0*sin(theta)) );
		res = vec(lnR);
	}

	return res;
}

#ifdef __CUDACC__

	template<typename OTHER>
	inline void image_copy(Ref & to, const OTHER & from, uint size) {
		to.data = from.data;
	}

	inline void image_copy(Host & to, const Host & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
	}

	inline void image_copy(Host & to, const Device & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyDeviceToHost);
	}

	inline void image_copy(Host & to, const HostDevice & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
	}

	inline void image_copy(Device & to, const Ref & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyDeviceToDevice);
	}

	inline void image_copy(Device & to, const Host & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToDevice);
	}

	inline void image_copy(Device & to, const Device & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyDeviceToDevice);
	}

	inline void image_copy(Device & to, const HostDevice & from, uint size) {
		cudaMemcpy(to.data, from.getDevice(), size, cudaMemcpyDeviceToDevice);
	}

	inline void image_copy(HostDevice & to, const Host & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
	}

	inline void image_copy(HostDevice & to, const Device & from, uint size) {
		cudaMemcpy(to.getDevice(), from.data, size, cudaMemcpyDeviceToDevice);
	}

	inline void image_copy(HostDevice & to, const HostDevice & from, uint size) {
		cudaMemcpy(to.data, from.data, size, cudaMemcpyHostToHost);
	}

	bool __forceinline__ operator==(const TrackData &d1,const TrackData &d2)
	{
		return d1.result==d2.result;
	}

	inline void synchroniseDevices()
	{
		cudaDeviceSynchronize();
	}

	inline float dist(const float3 &p1,const float3 &p2)
	{
		float3 p;
		p.x=p1.x-p2.x;
		p.y=p1.y-p2.y;
		p.z=p1.z-p2.z;
		return l2(p);
	}
#endif

sMatrix4 inverse(const sMatrix4 & A);

sMatrix6 inverse(const sMatrix6 & A);

float2 checkPoseErr(sMatrix4 p1,sMatrix4 p2);
#endif // UTILS_H
