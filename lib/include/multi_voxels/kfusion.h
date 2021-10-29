#ifndef ICS_FUSION_H
#define ICS_FUSION_H

#include"kparams.h"
#include"utils.h"
#include"volume.h"
#include"default_parameters.h"
#include<vector>
#include<iostream>

void checkMemAlloc (void * ptr);

void bilateralFilterKernel(float* out, const float* in, uint2 inSize,
const float * gaussian, float e_d, int r);

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
const float e_d, const int r);

void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize,
const sMatrix4 invK);
void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize);

void reduceKernel(float * out, TrackData* J, const uint2 Jsize,
const uint2 size);

void trackKernel(TrackData* output, const float3* inVertex,
const float3* inNormal, uint2 inSize, const float3* refVertex,
const float3* refNormal, uint2 refSize, const sMatrix4 Ttrack,
const sMatrix4 view, const float dist_threshold,
const float normal_threshold);

void integrateKernel(Volume vol, const float* depth, const uchar3* rgb, uint2 depthSize,
const sMatrix4 invTrack, const sMatrix4 K, const float mu,
const float maxweight, const int ratio);

void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
const Volume integration, const sMatrix4 view, const float nearPlane,
const float farPlane, const float step, const float largestep);

void initVolumeKernel(Volume volume);


class KFusion
{
	public:
	  
		//Allow a kfusion object to be created with a pose which include orientation as well as position
		KFusion(Configuration config/*const kparams_t &par, sMatrix4 initPose*/);

		~KFusion();
		
		// bool preprocessing(const ushort * inputDepth,const uchar3 *rgb);
		bool preprocessing(float *inputDepth,const uchar3 *rgb) ;

		bool tracking(uint frame);
		bool raycasting(uint frame);
		bool integration(uint frame);

		sMatrix4 getPose() const
		{
			return pose;
		}

		void setPose(const sMatrix4 pose_)
		{
			pose=pose_;
			forcePose=true;
			_tracked=true;
		}
		void setViewPose(sMatrix4 *value = NULL)
		{
			if (value == NULL)
				viewPose = &pose;
			else
				viewPose = value;
		}
		
		sMatrix4 *getViewPose()
		{
			return (viewPose);
		}

		Volume& getVolume()
		{
			return volume;
		}

		Volume getKeyFrameVolume()
		{
			return keyFrameVol;
		}

		void setKeyFramePose(int idx, const sMatrix4 &p)
		{
			volumes[idx].pose=p;
		}
		
		sMatrix4 getKeyFramePose(int idx) const
		{
			return volumes[idx].pose;
		}
		
		sMatrix4 getLastKFPose() const
		{
			return  lastKeyFramePose;
		}
		
		void integrateKeyFrameData();

		bool initKeyFrame(uint frame);

		// void saveVolumes(char *dir);
		bool fuseVolumes();
		
		uint keyFramesNum() const
		{
			return volumes.size();
		}
		
		// void clearKeyFramesData();
		void dropKeyFrame(int val);
		
		bool fuseLastKeyFrame(sMatrix4 &pose);
		kparams_t params;
		
		float *rawDepth;
		uchar3 *rawRgb;
		int _frame;
		bool _tracked;
		bool _integrated;
		int lastFrame;
	private:
		bool forcePose;
		float step;
		sMatrix4 pose;

		sMatrix4 oldPose;
		sMatrix4 deltaPose;
		sMatrix4 trackPose;

		sMatrix4 lastKeyFramePose;
		uint lastKeyFrameIdx;
		
		sMatrix4 *viewPose;
		sMatrix4 inverseCam;
		sMatrix4 camMatrix;
		std::vector<int> iterations;
		Volume volume;
		float largestep;
		Volume keyFrameVol;
		Volume fusionVol;
		int lastKeyFrame;


		sMatrix4 raycastPose;

		TrackData *reduction;
		float3 *vertex;
		float3 *normal;
		float3 **inputVertex;
		float3 **inputNormal;
		float **scaledDepth;

		float *output;
		float *gaussian;

		uchar3 *renderModel;

		std::vector<VolumeCpu> volumes;

		
		//Functions
		bool updatePoseKernel(sMatrix4 & pose, const float * output,float icp_threshold,sMatrix4 &deltaPose);
		bool checkPoseKernel(sMatrix4 & pose,
							 sMatrix4 oldPose,
							 const float * output,
							 uint2 imageSize,
							 float track_threshold);
};
#endif
