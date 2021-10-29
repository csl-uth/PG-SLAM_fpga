#include "g2o_slam/g2o_slam3d.hpp"


g2o_slam3d::g2o_slam3d(Configuration config)
	: exit(false)
{
	key_frame = 0;
	
	// create the linear solver
	auto linearSolver = g2o::make_unique<LinearSolverCSparse<BlockSolverX::PoseMatrixType>>();

	// create the block solver on top of the linear solver
	auto blockSolver = g2o::make_unique<BlockSolverX>(std::move(linearSolver));
	OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(std::move(blockSolver));

	optimizer.setAlgorithm(solver);

	vidx = -1;
	idx = -1;
	oidx = -1;

	max_depth = config.max_depth;
	min_depth = config.min_depth;
	max_num_kfs = config.max_num_kfs;
	max_num_fts = config.max_num_fts;
	knn_match_ratio = config.knn_match_ratio;
	min_num_matches = config.min_num_matches;
	g2o_max_iter = config.g2o_max_iter;
	g2o_ftsWeight = config.g2o_ftsWeight;
	image_freq = config.image_freq;
	odom_freq = config.odom_freq;
	
	freq = fmax(image_freq, odom_freq);
	std::vector<double> affine_list = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0 ,1 ,0, 0, 0, 0, 1};
	
	if(affine_list.size() == 16)
	{
		T_B_P(0, 0) = affine_list[0];
		T_B_P(0, 1) = affine_list[1];
		T_B_P(0, 2) = affine_list[2];
		T_B_P(0, 3) = affine_list[3];
		T_B_P(1, 0) = affine_list[4];
		T_B_P(1, 1) = affine_list[5];
		T_B_P(1, 2) = affine_list[6];
		T_B_P(1, 3) = affine_list[7];
		T_B_P(2, 0) = affine_list[8];
		T_B_P(2, 1) = affine_list[9];
		T_B_P(2, 2) = affine_list[10];
		T_B_P(2, 3) = affine_list[11];
		T_B_P(3, 0) = affine_list[12];
		T_B_P(3, 1) = affine_list[13];
		T_B_P(3, 2) = affine_list[14];
		T_B_P(3, 3) = affine_list[15];
	}
	else
	{
		T_B_P.Identity();
	}
	
	q_B_P = Quaterniond(T_B_P.linear());
	
	//add the camera parameters, caches are automatically resolved in the addEdge calls
	g2o::ParameterSE3Offset* cameraOffset = new g2o::ParameterSE3Offset;
	cameraOffset->setId(0);
	optimizer.addParameter(cameraOffset);
	
	//Initialize graph with an Identity Affine TF
	addPoseVertex(Eigen::Affine3d::Identity(),true);//Initial Pose is anchored 
	
	std::cout << "g2o initialization completed." << std::endl;
}


void g2o_slam3d::processingThread(PoseT odom_msg)
{
	static cv::Mat currImage, prevImage, currDepthImage, prevDepthImage;
	  
	ImageData img = image_data.pop();
	
	std::cout << "Key frame:" << img.frame << " " << std::endl;
	
	if(firstTime)
	{
		std::cout<<"First time"<<std::endl;
		if (img.rgb.channels() == 3)
		{
			cvtColor(img.rgb, currImage, cv::COLOR_BGR2GRAY);
		}
		else
		{
			currImage = img.rgb.clone();
		}
		currDepthImage = img.depth.clone();
		firstTime = false;
	}
	else
	{
		vector<cv::Point2f> pts1, pts2;
		vector<cv::DMatch> corr;
		vector<cv::KeyPoint> kpts1, kpts2;
	 
		cv::swap(currImage, prevImage);
		cv::swap(currDepthImage, prevDepthImage);
		
		cvtColor(img.rgb, currImage, cv::COLOR_BGR2GRAY);
		currDepthImage = img.depth.clone();
		
		if (findCorrespondingPoints(prevImage, currImage, kpts1, kpts2, pts1, pts2, corr) == false)
		{
			std::cout << "no matches found" << std::endl;  
			
			Eigen::Affine3d odom_pose = rosOdomToAffine(odom_msg);
			Eigen::Affine3d pose_0 = getPoseVertex(vidx);
			Eigen::Affine3d rel_odom_pose = pose_0.inverse() * odom_pose;
			addPoseVertex(odom_pose,false);
			addPoseEdge(rel_odom_pose, Eigen::Matrix<double, 6, 6>::Identity(), vidx);
		}
		else
		{
			Eigen::Affine3d odom = rosOdomToAffine(odom_msg);
			addMatchesToGraph(corr,odom,pts1,pts2,prevDepthImage,currDepthImage,true);
		}
	}

}

void g2o_slam3d::addMatchesToGraph(const vector<cv::DMatch> &corr,
								   Eigen::Affine3d &odom_pose,
								   const vector<cv::Point2f> &pts1,
								   const vector<cv::Point2f> &pts2,
								   const cv::Mat &prevDepthImage,
								   const cv::Mat &currDepthImage,
								   bool odom_inc)
{    
	std::cout << "Found " << corr.size() << " matches" << std::endl;

	Eigen::Affine3d pose_0 = getPoseVertex(vidx);
	if(odom_inc)
	{
		Eigen::Affine3d rel_odom_pose = pose_0.inverse() * odom_pose;
		addPoseVertex(odom_pose,false);
		addPoseEdge(rel_odom_pose, Eigen::Matrix<double, 6, 6>::Identity(), vidx);

	}
	else
	{
		addPoseVertex(Eigen::Affine3d::Identity(), false); //Unknown pose;
	}

	Eigen::Affine3d pose_1  =  getPoseVertex(vidx);
	Eigen::Vector3d pos1, rel_pos1, rel_pos0;
	// edges == factors
	for (unsigned int i = 0; i < corr.size(); i++)
	{
		rel_pos0 = projectuvXYZ(pts1[i], prevDepthImage);
		rel_pos1 = projectuvXYZ(pts2[i], currDepthImage);
		pos1 = pose_1 * rel_pos1;
	
		addObservationVertex(pos1,true);
		Eigen::Matrix3d RandomPose;
		addObservationEdges(rel_pos0, Eigen::Matrix3d::Identity()*g2o_ftsWeight, vidx-1, oidx);
		addObservationEdges(rel_pos1, Eigen::Matrix3d::Identity()*g2o_ftsWeight, vidx, oidx);            
	}
}

Eigen::Affine3d g2o_slam3d::rosOdomToAffine(const PoseT odom) const
{
	Eigen::Vector3d t_ = Vector3d(odom.position[0], odom.position[1], odom.position[2]);
	Quaterniond q_ = Quaterniond(odom.orientation[3], odom.orientation[0], odom.orientation[1], odom.orientation[2]);
	
	t_ = T_B_P*t_;
	q_ = q_B_P* q_ * q_B_P.inverse();
	Eigen::Affine3d odom_pose;
	odom_pose.translation() = t_;
	odom_pose.linear() = q_.toRotationMatrix();
	
	return odom_pose;
}


bool g2o_slam3d::findCorrespondingPoints(const cv::Mat &img1, 
										const cv::Mat &img2, 
										vector<cv::KeyPoint> &kp1, 
										vector<cv::KeyPoint> &kp2, 
										vector<cv::Point2f> &points1, 
										vector<cv::Point2f> &points2, 
										vector<cv::DMatch> &matches)
{
	cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create(max_num_fts);

	cv::Mat desp1, desp2;
	fdetector->detectAndCompute(img1, cv::Mat(), kp1, desp1);
	fdetector->detectAndCompute(img2, cv::Mat(), kp2, desp2);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

	vector<vector<cv::DMatch>> matches_knn;
	matcher->knnMatch(desp1, desp2, matches_knn, 2);
	for (size_t i = 0; i < matches_knn.size(); i++)
	{
		if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance)
			matches.push_back(matches_knn[i][0]);
	}

	if (int(matches.size()) <= min_num_matches) 
		return false;

	for (auto m : matches)
	{
		points1.push_back(kp1[m.queryIdx].pt);
		points2.push_back(kp2[m.trainIdx].pt);
	}

	return true;
}

void g2o_slam3d::imageDepthCb(int frame,
							  uchar3* rgb_msg,
							  float* depth_msg,
							  bool keyFrame,
							  uint2 size)
{    
	   
	if(!keyFrame)
		return;
	
	ImageData data;

	data.depth = cv::Mat(size.y, size.x, CV_32FC1, depth_msg);
	data.rgb = cv::Mat(size.y, size.x, CV_8UC3, rgb_msg);
	
	data.frame = frame;
	image_data.push(data);       
}


void g2o_slam3d::cameraInfoCb(Configuration config)
{
		height = 480; //msg->height;
		width = 640; //msg->width;

		k1 = 0.0;
		k2 = 0.0;
		t1 = 0.0;
		t2 = 0.0;
		k3 = 0.0;

		fx = config.camera.x;
		cx = config.camera.y;
		fy = config.camera.z;
		cy = config.camera.w;
}

int g2o_slam3d::getObservationVertexId(int oidx_)
{
	return  oidx_map[oidx_];
}

int g2o_slam3d::getPoseVertexId(int vidx_)
{
	return  vidx_map[vidx_];
}


void g2o_slam3d::addPoseVertex(Eigen::Affine3d pose_, bool isFixed = false)
{

	g2o::VertexSE3 *v = new g2o::VertexSE3();
	vidx++;
	idx++;
	vidx_map[vidx]=idx;
	v->setId(idx);
	v->setFixed(isFixed);
	v->setEstimate(g2o::SE3Quat(pose_.linear(),pose_.translation()));
	optimizer.addVertex(v);
	getPoseVertex(vidx);
}

void g2o_slam3d::addObservationVertex(Eigen::Vector3d pos_, bool isMarginalized = true)
{
		g2o::VertexPointXYZ* v = new g2o::VertexPointXYZ();
		oidx++;
		idx++;
		oidx_map[oidx]=idx;
		v->setId(idx);
		v->setEstimate(pos_);
		v->setMarginalized(isMarginalized);
		optimizer.addVertex(v);

}

Eigen::Vector3d g2o_slam3d::projectuvXYZ(cv::Point2f pts, cv::Mat depthImg)
{
	Eigen::Vector3d ret = Eigen::Vector3d::Zero();
	//Pinhole model to set Initial Point Estimate
	double z = 1.0;
	int uu = cvRound(pts.x);
	int vv = cvRound(pts.y);
	if ((uu < width && uu >= 0 && vv >= 0 && vv < height))
	{
		z = depthImg.at<float>(vv, uu);
		if (z < min_depth || z > max_depth || z != z)
			z = 1.0;
	}

	ret(0) = (pts.x - cx) * z / fx;
	ret(1) = (pts.y - cy) * z / fy;
	ret(2) = z;

	return ret;
}

void g2o_slam3d::addPoseEdge(Eigen::Affine3d pose, Eigen::Matrix<double, 6, 6> cov, int vertexId)
{
	//add odometry edge
	g2o::EdgeSE3 *odom=new g2o::EdgeSE3();
	// get two poses
	VertexSE3* vp0 = dynamic_cast<VertexSE3*>(optimizer.vertices().find(getPoseVertexId(vertexId-1))->second);
	VertexSE3* vp1 = dynamic_cast<VertexSE3*>(optimizer.vertices().find(getPoseVertexId(vertexId))->second);
	odom->setVertex(0,vp0);
	odom->setVertex(1,vp1);
	odom->setInformation(cov);
	odom->setParameterId(0, 0);
	odom->setMeasurement(g2o::SE3Quat(pose.linear(),pose.translation()));
	optimizer.addEdge(odom);
}

// edges == factors
void g2o_slam3d::addObservationEdges(Eigen::Vector3d p, Eigen::Matrix3d cov, int vertexId, int obsId)
{
		g2o::EdgeSE3PointXYZ *edge = new g2o::EdgeSE3PointXYZ();
		VertexPointXYZ* vp0 = dynamic_cast<VertexPointXYZ*>(optimizer.vertices().find(getObservationVertexId(obsId))->second);
		VertexSE3* vp1 = dynamic_cast<VertexSE3*>(optimizer.vertices().find(getPoseVertexId(vertexId))->second);
		edge->setVertex(0, vp1);
		edge->setVertex(1, vp0);
		edge->setMeasurement(p);
		edge->setInformation(cov);
		edge->setParameterId(0, 0);
		edge->setRobustKernel(new g2o::RobustKernelHuber());
		optimizer.addEdge(edge);
		edges.push_back(edge);
}

void g2o_slam3d::solve(int num_iter = 10, bool verbose = false)
{
	optimizer.setVerbose(verbose);
	optimizer.initializeOptimization();
	optimizer.optimize(num_iter);
}

Eigen::Affine3d g2o_slam3d::getPoseVertex(int vertexId)
{
	int tmp_id = getPoseVertexId(vertexId);
	g2o::VertexSE3 *v = dynamic_cast<g2o::VertexSE3 *>(optimizer.vertex(tmp_id));
	Eigen::Isometry3d tmp_pose = v->estimate();
	Eigen::Affine3d pose = Eigen::Affine3d::Identity();
	pose.linear() = tmp_pose.linear();
	pose.translation() = tmp_pose.translation();
	//cout << "Pose=" << endl << pose.matrix() << endl;
	return pose;
}

Eigen::Vector3d g2o_slam3d::getObservationVertex(int obsId)
{
	int tmp_id = getObservationVertexId(obsId);
	g2o::VertexPointXYZ *v = dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(tmp_id));
	Eigen::Vector3d pos = v->estimate();

	return pos;
}

void g2o_slam3d::getInliers()
{
	int inliers = 0;
	for (auto e : edges)
	{
		e->computeError();
		// chi2 error > Pixel uncertainty
		if (e->chi2() > 1.5)
		{
			//cout << "error = " << e->chi2() << endl;
		}
		else
		{
			inliers++;
		}
	}
	cout << "inliers in total points: " << inliers << "/" << edges.size() << endl;
}

PathT g2o_slam3d::optimize()
{
	solve(g2o_max_iter, true); //10 iterations in G2O and verbose
	cout << " NUM OF POSE VERTICES " << vidx << endl;
	cout << " NUM OF LANDMARK VERTICES " << oidx << endl;
	
	PathT opt_odom_path_msg;
	// nav_msgs::Odometry opt_pose_msg;
	Eigen::Quaterniond opt_pose_q;
	Eigen::Affine3d opt_pose;
	// Eigen::Vector3d opt_point;
	
	// sensor_msgs::PointCloud pt_msg;
	#ifdef G2O_OPT_MAX_NUM_KFS
	for (int i = vidx - max_num_kfs; i >= 0 && i <= vidx; i++)
	#else
	for (int i = 0; i <= vidx; i++)
	#endif
	{
		opt_pose = getPoseVertex(i);
		opt_pose_q = Eigen::Quaterniond(opt_pose.linear());

		PoseT tmp_pose;

		tmp_pose.position[0] = opt_pose.translation()(0);
		tmp_pose.position[1] = opt_pose.translation()(1);
		tmp_pose.position[2] = opt_pose.translation()(2);
		tmp_pose.orientation[0] = opt_pose_q.x();
		tmp_pose.orientation[1] = opt_pose_q.y();
		tmp_pose.orientation[2] = opt_pose_q.z();
		tmp_pose.orientation[3] = opt_pose_q.w();
		opt_odom_path_msg.poses.push_back(tmp_pose);
	}

	return opt_odom_path_msg;
}
