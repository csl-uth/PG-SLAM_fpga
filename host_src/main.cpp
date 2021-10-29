/* parameters : 
 *	- key frame rate: 5
 *	- camera info (height, width, D, K)
 *	- compute_size_ratio: 2
 *	- integration_rate: 1
 *	- rendering_rate: 1
 *	- tracking_rate: 1
 *	- volume_resolution: [256,256,256]
 *	- volume_direction: [4.0,4.0,4.0]
 *	- volume_size: [8,8,8]
 *	- pyramid: [10,5,4]
 *	- mu: 0.1
 *	- icp_threshold: 5.0e-01
 *	- max depth: 4.0
 *	- min depth: 0.01
 *	- 
 */
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string.h>
#include <g2o_slam/g2o_slam3d.hpp>

#include<kparams.h>
#include<kfusion.h>
#include<volume.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std; 


inline double tock() {
		struct timespec clockData;
		clock_gettime(CLOCK_MONOTONIC, &clockData);

		return (double) clockData.tv_sec + clockData.tv_nsec / 1000000000.0;
}

KFusion *fusion = nullptr;

FILE* pFile ;

int frame = 0;

int keyframe = 0;

bool isKeyFrame = false;

float *depthIn;
uchar3 *rgbIn;
uint16_t *inputDepth = NULL;
float *inputDepthFl = NULL;


std::ostream* logstream = &std::cout;
std::ofstream logfilestream;

// functions declaration
bool readNextRGBDimage (uint2 *newImageSize);
void publishOdom(PoseT *curr_pose);

sMatrix4 homoFromRosPose(const PoseT &p);

void optimizedPathCb(const PathT msg);

double varianceOfLaplacian(const cv::Mat& src);


int main(int argc, char **argv)
{
	// to do: parameterize input parameters
	Configuration config(argc, argv);

	if (config.log_file != "") {
		logfilestream.open(config.log_file);
		logstream = &logfilestream;
	}
	if (config.input_file == "") {
		std::cerr << "No input found." << std::endl;
		config.print_arguments();
		exit(1);
	}

	pFile = fopen(config.input_file.c_str(), "rb");
	g2o_slam3d *g2o_node;
	g2o_node = new g2o_slam3d(config);

	fusion = new KFusion(config);

	g2o_node->cameraInfoCb(config);

	uint2 newImageSize;
	uint2 size = fusion->params.inputSize;

	depthIn = (float*) malloc(size.x * size.y * sizeof(float));
	checkMemAlloc(depthIn);
	rgbIn = (uchar3*) malloc(size.x * size.y * sizeof(uchar3));
	checkMemAlloc(rgbIn);

	double timings[9];
	timings[0] = tock();

	*logstream
			<< "frame\tacquisition\tpreprocessing\ttracking\tintegration\traycasting\tkfusion-total\tg2o\ttotal    \tX          \tY          \tZ          \ttracked \tintegrated\tstartTS\tendTS"
			<< std::endl;
	logstream->setf(std::ios::fixed, std::ios::floatfield);

	while(readNextRGBDimage(&newImageSize)) {
		std::cerr << "["<< frame << "] \r" << std::flush;

		// check if it is a key frame 
		if ((config.kf_rate != 0) && (frame % config.kf_rate == 0)) {
			cv::Mat rgb_mat (newImageSize.y, newImageSize.x, CV_8UC3, rgbIn);
			double focusMeasure = varianceOfLaplacian(rgb_mat);
			
			if (focusMeasure >= config.blurr_threshold) {
				isKeyFrame = true;
			}
		}

		// multi-voxels part
		
		fusion->_frame++;
		fusion->lastFrame = frame;

		sMatrix4 tempPose = fusion->getPose();
		float xt = tempPose.data[0].w;
		float yt = tempPose.data[1].w;
		float zt = tempPose.data[2].w;

		
		timings[1] = tock();

		fusion->preprocessing(depthIn, rgbIn);

		timings[2] = tock();

		if (frame % fusion->params.tracking_rate == 0)
			fusion->_tracked = fusion->tracking(frame);

		timings[3] = tock();

		if(isKeyFrame)
		{
			fusion->initKeyFrame(frame);
			#ifdef PREF_FUSEVOL
				if(frame != 0)
					fusion->prefetch_kfvol_flag = frame;
			#endif
		}

		
		timings[4] = tock();

		if (frame % fusion->params.integration_rate == 0)
			fusion->_integrated = fusion->integration(frame);
		
		timings[5] = tock();
		
		if(fusion->_integrated) 
		{
			fusion->integrateKeyFrameData();
		}

		timings[6] = tock();

		#ifndef R_INTERLEAVING
			#ifdef R_RATE
			if (frame % RAYCAST_RATE == 3)
			#endif
			fusion->raycasting(frame);
		#endif

		timings[7] = tock();

		// publish multi-voxels odometry to g2o
		PoseT curr_pose;
		publishOdom(&curr_pose);

		if (isKeyFrame) {
			// g2o part
			g2o_node->imageDepthCb(frame, rgbIn, depthIn, isKeyFrame, size);	// to do: need arguments
			g2o_node->processingThread(curr_pose);

			if (keyframe > 0 && keyframe % g2o_node->max_num_kfs == 0) {
				PathT opt_path = g2o_node->optimize();
				optimizedPathCb(opt_path);
			}
			keyframe++;
		}
		
		timings[8] = tock();
		
		*logstream  << frame << "\t\t" << timings[1] - timings[0] << "\t" //  acquisition
					<< timings[2] - timings[1] << "\t"     //  preprocessing
					<< timings[3] - timings[2] << "\t"     //  tracking
					<< (timings[5] - timings[4]) + (timings[6] - timings[5]) << "\t"     //  integration
					<< timings[7] - timings[6] << "\t"     //  raycasting
					<< (timings[7] - timings[1]) - (timings[4] - timings[3]) << "\t"     //  kfusion-total
					<< (timings[8] - timings[7]) + (timings[4] - timings[3]) << "\t"     //  g2o-keyFrame
					<< timings[8] - timings[0] << "\t"     //  total
					<< xt << "\t" << yt << "\t" << zt << "\t"     //  X,Y,Z
					<< fusion->_tracked << "        \t" << fusion->_integrated  << "\t" // tracked and integrated flags
					<< timings[0] << "\t" // frame start timestamp
					<< timings[8] << "\t" // frame end timestamp
					<< std::endl;

		frame++;
		isKeyFrame = false;

		timings[0] = tock();
	}

	fusion->~KFusion();
	free(rgbIn);
	
	return 0;
}


bool readNextRGBDimage (uint2 *newImageSize) {

	if (!fread(newImageSize, sizeof(uint2), 1, pFile)) {
		cout << "EOF" << endl;
		fclose(pFile);
		return false;
	}

	uint read_bytes = 0;
	do {
		read_bytes = fread(&depthIn[read_bytes], sizeof(float), (newImageSize->x * newImageSize->y) - read_bytes, pFile);
	}
	while (read_bytes != newImageSize->x * newImageSize->y);

	if (!fread(newImageSize, sizeof(uint2), 1, pFile)) {
		cout << "Error reading file" << endl;
		fclose(pFile);
		return false;
	}
	
	read_bytes = 0;
	do {
		read_bytes = fread(&rgbIn[read_bytes], sizeof(unsigned char), (3 * newImageSize->x * newImageSize->y)-read_bytes, pFile);
	}
	while (read_bytes != 3*newImageSize->x * newImageSize->y);

	return true;
}


void publishOdom(PoseT *curr_pose)
{
	sMatrix4 pose = fusion->getPose();

	pose(0,3) -= fusion->params.volume_direction.x;
	pose(1,3) -= fusion->params.volume_direction.y;
	pose(2,3) -= fusion->params.volume_direction.z;

	Eigen::Matrix3f rot_matrix;
	rot_matrix(0,0) = pose(0,0);
	rot_matrix(0,1) = pose(0,1);
	rot_matrix(0,2) = pose(0,2);
	rot_matrix(1,0) = pose(1,0);
	rot_matrix(1,1) = pose(1,1);
	rot_matrix(1,2) = pose(1,2);
	rot_matrix(2,0) = pose(2,0);
	rot_matrix(2,1) = pose(2,1);
	rot_matrix(2,2) = pose(2,2);

	Eigen::Quaternionf q (rot_matrix);

	curr_pose->position[0] = pose(0,3);
	curr_pose->position[1] = pose(1,3);
	curr_pose->position[2] = pose(2,3);
	curr_pose->orientation[0] = q.x();
	curr_pose->orientation[1] = q.y();
	curr_pose->orientation[2] = q.z();
	curr_pose->orientation[3] = q.w(); 
}

sMatrix4 homoFromRosPose(const PoseT &p)
{
	sMatrix4 ret;
	Eigen::Quaternionf q;
	q.x() = p.orientation[0];
	q.y() = p.orientation[1];
	q.z() = p.orientation[2];
	q.w() = p.orientation[3];
	
	
	Eigen::Matrix3f rot(q);
	
	for(int i=0; i < 3; i++)
	{
		for(int j=0; j < 3; j++)
		{
			 ret(i,j) = rot(i,j);
		}
	}
	ret(0,3) = p.position[0];
	ret(1,3) = p.position[1];
	ret(2,3) = p.position[2];
	ret(3,3) = 1;
	return ret;
	
}

void optimizedPathCb(const PathT msg)
{
	std::cout << "Got optimized poses" << std::endl;
	int msgPosesSize = msg.poses.size();
	int keyFramesSize = fusion->keyFramesNum();
	sMatrix4 lastKFPose;
	sMatrix4 currentPose;

	if(msgPosesSize > keyFramesSize+1) {
		std::cout << "Got more poses than key frames" << std::endl;
		std::cout << keyFramesSize+1 << " poses is expected but got " << msgPosesSize << std::endl;
		return ;
	}

	//The last optimized pose is the pose of the current key frame
	if(msgPosesSize == keyFramesSize+1) {
		for(int i=0; i < msgPosesSize-1; i++) {
			PoseT poseRos = msg.poses[i];
			sMatrix4 optPose = homoFromRosPose(poseRos);
			
			optPose(0,3) += fusion->params.volume_direction.x;
			optPose(1,3) += fusion->params.volume_direction.y;
			optPose(2,3) += fusion->params.volume_direction.z;                
			
			fusion->setKeyFramePose(i, optPose);
		}

		PoseT lastOptPoseRos = msg.poses.back();
		lastKFPose = homoFromRosPose(lastOptPoseRos);
		
		lastKFPose(0,3) += fusion->params.volume_direction.x;
		lastKFPose(1,3) += fusion->params.volume_direction.y;
		lastKFPose(2,3) += fusion->params.volume_direction.z;
		
		sMatrix4 lastKFPoseKF = fusion->getLastKFPose();
		sMatrix4 currKPoseKF = fusion->getPose();

		sMatrix4 delta = inverse(lastKFPoseKF) * currKPoseKF;
		currentPose = lastKFPose*delta;
	}
	//The last optimized pose is older than the current key frame
	else
	{ 
		std::cout << "Got " << msgPosesSize << "optimized poses but " << keyFramesSize+1 << " is expected." << std::endl;

		//kfp is the corresponding pose of kfusion to the last optimized pose
		sMatrix4 kfp = fusion->getKeyFramePose(msgPosesSize-1);
		sMatrix4 optPose;
		for(int i=0; i < msgPosesSize; i++)
		{
			PoseT poseRos = msg.poses[i];
			optPose = homoFromRosPose(poseRos);
			
			optPose(0,3) += fusion->params.volume_direction.x;
			optPose(1,3) += fusion->params.volume_direction.y;
			optPose(2,3) += fusion->params.volume_direction.z;                
			
			fusion->setKeyFramePose(i, optPose);
		}
					
		sMatrix4 delta=optPose * inverse(kfp);

		for(int i=msgPosesSize; i < keyFramesSize; i++)
		{
			sMatrix4 prevPose = fusion->getKeyFramePose(i);
			sMatrix4 newPose = delta*prevPose;
			fusion->setKeyFramePose(i, newPose);
		}    
		
		lastKFPose=delta*fusion->getLastKFPose();
		currentPose=delta*fusion->getPose();
	}    

	std::cout << "Fusing volumes" << std::endl;
	fusion->fuseVolumes();

	#ifndef NO_FUSELASTKF
		fusion->fuseLastKeyFrame(lastKFPose);
	#endif

	fusion->setPose(currentPose);
	fusion->raycasting(frame);

}

double varianceOfLaplacian(const cv::Mat& src)
{
    cv::Mat lap;
    cv::Laplacian(src, lap, CV_64F);

    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);

    double focusMeasure = sigma.val[0]*sigma.val[0];
    return focusMeasure;
}