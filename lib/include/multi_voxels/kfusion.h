#ifndef ICS_FUSION_H
#define ICS_FUSION_H

#include"kparams.h"
#include"utils.h"
#include"volume.h"
#include"default_parameters.h"
#include<vector>
#include<iostream>

#define NUM_LEVELS 3
#define KFUSION_INVALID -2

#ifdef INT_NCU
#define CU 4
#endif

#ifdef INT_LP4
#define LOOP_STEP 4
#else
#ifdef INT_LP6
#define LOOP_STEP 6
#endif
#endif

#ifdef R_RATE
#define RAYCAST_RATE 4
#endif

#ifdef PREF_FUSEVOL
#define MAX_NUM_KF 5
#endif

void checkMemAlloc (void * ptr);

#ifndef BF_HW
void bilateralFilterKernel(float* out, const float* in, uint2 inSize,
const float * gaussian, float e_d, int r);
#endif

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
		
		bool preprocessing(float *inputDepth,const uchar3 *rgb) ;

		bool tracking(uint frame);
		bool raycasting(uint frame);
		bool integration(uint frame);
		void integrateKernel(Volume vol, const float* depth, const uchar3* rgb, uint2 depthSize,
			const sMatrix4 invTrack, const sMatrix4 K, const float mu,
			const float maxweight, const int ratio);
		void integrateHWKernel(Volume vol, const float* depth, const uchar3* rgb, uint2 depthSize,
		const sMatrix4 invTrack, const sMatrix4 K, const float mu,
		const float maxweight, const int ratio);
		
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

		bool fuseVolumes();
		
		uint keyFramesNum() const
		{
			return volumes.size();
		}
		
		void dropKeyFrame(int val);
		void vitisRelease(void);
		
		bool fuseLastKeyFrame(sMatrix4 &pose);
		kparams_t params;
		
		float *rawDepth;
		uchar3 *rawRgb;
		int _frame;
		bool _tracked;
		bool _integrated;
		int lastFrame;

		#ifdef PREF_FUSEVOL
			int prefetch_kfvol_flag;
		#endif

		#ifdef POSEGRAPHOPT_ENABLED
			int posegraphOptEnabled;
		#endif
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

		#ifdef TR_HW
			float4 *reduction;
			float4 *vertex;
			float4 *normal;
			float4 **inputVertex;
			float4 **inputNormal;
		#else
			TrackData *reduction;
			float3 *vertex;
			float3 *normal;
			float3 **inputVertex;
			float3 **inputNormal;
		#endif

		float **scaledDepth;

		#ifdef INT_NCU
			uint4* inSize[CU];
			#ifdef INT_HP
				floatH4* inDim[CU];
				floatH4* inConst[CU];
			#else
				float4* inDim[CU];
				float4* inConst[CU];
			#endif // INT_HP
		#else
			uint4* inSize;
			#ifdef INT_HP
				floatH4* inDim;
				floatH4* inConst;
			#else
				float4* inDim;
				float4* inConst;
			#endif
		#endif  // INT_NCU

		short2 *integrate_vol_ptr;

		#ifdef INTKF 
			short2 *integrate_kf_vol_ptr;
			#ifdef INTKF_UNIQUEUE
			#ifdef INT_NCU
				uint4* inSize_kf[CU];
				#ifdef INT_HP
					floatH4* inDim_kf[CU];
					floatH4* inConst_kf[CU];
				#else
					float4* inDim_kf[CU];
					float4* inConst_kf[CU];
				#endif // INT_HP
			#else
				uint4* inSize_kf;
				#ifdef INT_HP
					floatH4* inDim_kf;
					floatH4* inConst_kf;
				#else
					float4* inDim_kf;
					float4* inConst_kf;
				#endif
			#endif  // INT_NCU
			#endif  // INTKF_UNIQUEUE
		#endif
		// Image<ushort, Device> depthImage;

		#ifdef PREF_FUSEVOL
			#ifdef FUSE_HP
			floatH *fuseVol_interp_ptr[MAX_NUM_KF];
			#else
			float *fuseVol_interp_ptr[MAX_NUM_KF];
			#endif
		#endif

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
