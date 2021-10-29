#include"utils.h"

#include<eigen3/Eigen/Dense>

struct timespec tick_clockData;
struct timespec tock_clockData;

sMatrix4 T_B_P(0, -1,  0, 0,
			   0,  0, -1, 0,
			   1,  0,  0, 0,
			   0,  0,  0, 1 );

sMatrix4 invT_B_P = inverse(T_B_P);

inline float sq(float r) {
	return r * r;
}

inline double sq(double r) {
	return r * r;
}

inline sMatrix3 operator*(const sMatrix3 & A, const sMatrix3 & B)
{
	sMatrix3 R;
	TooN::wrapMatrix<3, 3>(&R.data[0]) = TooN::wrapMatrix<3, 3>(&A.data[0])
			* TooN::wrapMatrix<3, 3>(&B.data[0]);
	return R;
}

inline sMatrix6 operator*(const sMatrix6 & A, const sMatrix6 & B)
{
	sMatrix6 R;
	TooN::wrapMatrix<6, 6>(&R.data[0]) = TooN::wrapMatrix<6, 6>(&A.data[0])
			* TooN::wrapMatrix<6, 6>(&B.data[0]);
	return R;
}

inline sMatrix6 operator*(const sMatrix6 & A, const float f)
{
	sMatrix6 R;
	for(int i=0;i<36;i++)
		R.data[i]=A.data[i]*f;
	return R;
}

inline sMatrix3 operator*(const sMatrix3 & A, const float f)
{
	sMatrix3 R;
	for(int i=0; i<9; i++)
		R.data[i] = A.data[i]*f;
	return R;
}

inline sMatrix3 wedge(float *v)
{
	sMatrix3 skew;
	for(int i=0; i<3*3; i++)
		skew.data[i] = 0.0;

	skew(0, 1) = -v[2];
	skew(0, 2) = v[1];
	skew(1, 2) = -v[0];
	skew(1, 0) = v[2];
	skew(2, 0) = -v[1];
	skew(2, 1) = v[0];

	return skew;

}

inline sMatrix3 transpose( const sMatrix3 &mat)
{
	sMatrix3 ret;
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			ret(j,i) = mat(i,j);
	return ret;
}

float2 checkPoseErr(sMatrix4 p1,sMatrix4 p2)
{
	float2 ret;
	sMatrix3 r1,r2;

	float3 tr1 = make_float3(p1(0,3),p1(1,3),p1(2,3));
	float3 tr2 = make_float3(p2(0,3),p2(1,3),p2(2,3));

	tr1 = tr1-tr2;
	ret.x = l2(tr1);

	for(int i=0; i < 3; i++)
	{
		for(int j=0; j < 3; j++)
		{
			r1(i,j) = p1(i,j);
			r2(i,j) = p2(i,j);
		}
	}
	r1 = r1*transpose(r2);
	float3 f = logMap(r1);

	ret.y = l2(f);
	return ret;
}

inline Eigen::MatrixXd toEigen(const sMatrix4 &mat)
{
	Eigen::MatrixXd ret(4,4);
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4; j++)
		{
			ret(i,j) = mat(i,j);
		}
	}
	return ret;
}

inline Eigen::MatrixXd toEigen(const sMatrix3 &mat)
{
	Eigen::MatrixXd ret(3,3);
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			ret(i,j) = mat(i,j);
		}
	}
	return ret;
}

inline Eigen::MatrixXd toEigen(const sMatrix6 &mat)
{
	Eigen::MatrixXd ret(6,6);
	for(int i=0; i<6; i++)
	{
		for(int j=0; j<6; j++)
		{
			ret(i,j) = mat(i,j);
		}
	}
	return ret;
}

inline sMatrix4 fromEigen4(const Eigen::MatrixXd &mat)
{
	sMatrix4 ret;
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<4; j++)
		{
			ret(i,j) = mat(i,j);
		}
	}
	return ret;
}

inline sMatrix6 fromEigen6(const Eigen::MatrixXd &mat)
{
	sMatrix6 ret;
	for(int i=0; i<6; i++)
	{
		for(int j=0; j<6; j++)
		{
			ret(i,j) = mat(i,j);
		}
	}
	return ret;
}

inline sMatrix3 fromEigen3(const Eigen::MatrixXd &mat)
{
	sMatrix3 ret;
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			ret(i,j) = mat(i,j);
		}
	}
	return ret;
}

sMatrix4 inverse(const sMatrix4 & A)
{
	static TooN::Matrix<4, 4, float> I = TooN::Identity;
	TooN::Matrix<4, 4, float> temp = TooN::wrapMatrix<4, 4>(&A.data[0].x);
	sMatrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = TooN::gaussian_elimination(temp, I);
	return R;

}

sMatrix6 inverse(const sMatrix6 & A)
{

	static TooN::Matrix<6, 6, float> I = TooN::Identity;
	TooN::Matrix<6, 6, float> temp = TooN::wrapMatrix<6, 6>(&A.data[0]);
	sMatrix6 R;
	TooN::wrapMatrix<6, 6>(&R.data[0]) = TooN::gaussian_elimination(temp, I);
	return R;

}
