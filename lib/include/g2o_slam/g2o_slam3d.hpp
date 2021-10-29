// for std
#include <iostream>
// for opencv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
// #ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
// #endif
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>


#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>

// #include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/edge_se3_pointxyz.h>
#include <g2o/types/slam3d/edge_se3.h>

#include <Eigen/Dense>

#include<vector_types.h>
#include<g2o_slam/Queue.h>
#include"default_parameters.h"


#include <fstream>
#include <map>
using namespace std;
using namespace g2o;
using namespace Eigen;


struct ImageData
{
	cv::Mat depth;
	cv::Mat rgb;
	int frame;
};

typedef struct 
{
	float position[3];
	float orientation[4];
} PoseT;

typedef struct 
{
	vector<PoseT> poses;
} PathT;

class g2o_slam3d
{
private:
	//g2o optimizer
	g2o::SparseOptimizer optimizer;
	vector<g2o::EdgeSE3PointXYZ *> edges;
	
	int height, width;    // image dimensions
	double k1, k2, k3, t1, t2;   // camera distorsion
	double cx, cy, fx, fy;	// camera calibration
	Queue<ImageData> image_data;        
	int key_frame;		//last frame send
	/// Flags for first Image Callback, first Camera Info Callback, and for new image callback
	std::map<int,int> oidx_map, vidx_map;
	
	int vidx, oidx, idx;
	double min_depth, max_depth;
	
	//max frequency
	double freq, image_freq, odom_freq;
	
	/// Topics
	std::string image_topic, depth_topic, cam_info_topic, odom_topic, key_frame_topic;
	
	Eigen::Affine3d T_B_P;
	Eigen::Quaterniond q_B_P;
	
	/// Functions
	Eigen::Affine3d rosOdomToAffine(const PoseT odom) const;
	
	bool findCorrespondingPoints(const cv::Mat &img1, 
								 const cv::Mat &img2,
								 vector<cv::KeyPoint> &kp1, 
								 vector<cv::KeyPoint> &kp2, 
								 vector<cv::Point2f> &points1, 
								 vector<cv::Point2f> &points2, 
								 vector<cv::DMatch> &matches);
	
	void addMatchesToGraph(const vector<cv::DMatch> &corr,
								   Eigen::Affine3d &odom_pose,
								   const vector<cv::Point2f> &pts1,
								   const vector<cv::Point2f> &pts2,
								   const cv::Mat &prevDepthImage,
								   const cv::Mat &currDepthImage,
								   bool odom_inc);
	
	bool exit;
	int  max_num_fts, min_num_matches, g2o_max_iter;
	double knn_match_ratio , g2o_ftsWeight;
public:
	int max_num_kfs;
	bool firstTime=true;

	PathT optimize();
	
	g2o_slam3d(Configuration config);
	
	void cameraInfoCb(Configuration config);        
	void imageDepthCb(int frame,
					  uchar3* rgb_msg,
					  float* depth_msg,
					  bool keyFrame,
					  uint2 size);

	
	/// Threads
	void processingThread(PoseT odom);
	void addPoseVertex(Eigen::Affine3d pose, bool isFixed);
	void addObservationVertex(Eigen::Vector3d pos_, bool isMarginalized);
	void addPoseEdge(Eigen::Affine3d pose, Eigen::Matrix<double, 6, 6> cov, int vertexId);
	// edges == factors
	void addObservationEdges(Eigen::Vector3d p, Eigen::Matrix3d cov, int vertexId, int obsId);

	void solve(int num_iter, bool verbose);

	Eigen::Affine3d getPoseVertex(int vertexId);

	Eigen::Vector3d getObservationVertex(int obsId);
	Eigen::Vector3d projectuvXYZ(cv::Point2f pts, cv::Mat depthImg);
	void getInliers();
	
	int getPoseVertexId(int vidx_);
	int getObservationVertexId(int oidx_);

};
