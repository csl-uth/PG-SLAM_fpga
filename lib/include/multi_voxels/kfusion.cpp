#include "kfusion.h"
#include <vector_types.h>
#include "constant_parameters.h"
#include "utils.h"
#include "volume.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>

#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <string.h>


#define KERNEL_TIMINGS
#ifdef KERNEL_TIMINGS
	FILE *kernel_timings_log;
#endif

bool print_kernel_timing = false;

using namespace std; 

#define TICK()    {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}}

#ifndef KERNEL_TIMINGS
#define TOCK(str,size)  {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tock_clockData); std::cerr<< str << " ";\
	if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
	std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}
#else
#define TOCK(str,size) { if(print_kernel_timing) { clock_gettime(CLOCK_MONOTONIC, &tock_clockData);\
							  fprintf(kernel_timings_log,"%s\t%d\t%f\t%f\n",str,size,(double) tick_clockData.tv_sec + tick_clockData.tv_nsec / 1000000000.0,(double) tock_clockData.tv_sec + tock_clockData.tv_nsec / 1000000000.0); }}
#endif

#ifdef PREF_FUSEVOL
#ifdef FUSE_HP
void prefetchFuseVolInterp (floatH *out, const Volume v, 
	const sMatrix4 pose, const float3 origin, const float3 voxelSize);
#else
void prefetchFuseVolInterp (float *out, const Volume v, 
	const sMatrix4 pose, const float3 origin, const float3 voxelSize);
#endif
#endif


// VITIS
#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl2.hpp>

cl_uint load_file_to_memory(const char *filename, char **result)
{
	cl_uint size = 0;
	FILE *f = fopen(filename, "rb");
	if (f == NULL) {
		*result = NULL;
		return -1; // -1 means file opening fail
	}
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size+1);
	if (size != fread(*result, sizeof(char), size, f)) {
		free(*result);
		return -2; // -2 means file reading fail
	}
	fclose(f);
	(*result)[size] = 0;
	return size;
}


cl_int err;
cl_uint check_status = 0;

cl_platform_id platform_id;       
cl_platform_id platforms[16];       
cl_uint platform_count;
cl_uint platform_found = 0;
char cl_platform_vendor[1001];

cl_uint num_devices;
cl_uint device_found = 0;
cl_device_id devices[16];  
char cl_device_name[1001];
cl_device_id device_id;
cl_context context;
cl_command_queue q;
#ifdef INTKF
	cl_command_queue q_intkf;
#endif
cl_command_queue q_tr;
cl_program program;

// bilateralFilter kernel variables
#ifdef BF_HW
cl_command_queue q_bf;

cl_kernel krnl_bilateralFilterKernel;

cl_mem floatDepth_buffer;
cl_mem scaledDepth_zero_buffer;
cl_mem gaussian_buffer;
cl_mem pt_bf[3];

size_t krnl_paddepth_size;
size_t krnl_gaussian_size;
size_t krnl_out_size;

float * pad_depth; // padded floatdeph
uint2  padSize;
#endif

// track kernel variables
#ifdef TR_HW
cl_kernel krnl_trackKernel;

cl_mem inVertex_buffer[NUM_LEVELS];
cl_mem inNormal_buffer[NUM_LEVELS];

cl_mem refVertex_buffer;
cl_mem refNormal_buffer;

cl_mem Ttrack_data_buffer;
float* Ttrack_data;
cl_mem view_data_buffer;
float* view_data;

cl_mem trackData_float_buffer;

cl_mem pt_tr_in[6];
cl_mem pt_tr_out[1];
#endif

cl_int status;

// integrate kernel variables
#ifdef INT_NCU
	cl_kernel krnl_integrateKernel[CU];

	cl_mem integrate_depth_buffer[CU];
	#ifdef INT_HP
		floatH *floatDepth[CU];
	#else
		float *floatDepth[CU];
	#endif
	cl_mem volSize_buffer[CU];
	cl_mem volDim_buffer[CU];
	cl_mem volConst_buffer[CU];

	cl_mem InvTrack_data_buffer[CU];
	#ifdef INT_HP
		floatH* InvTrack_data[CU];
	#else
		float* InvTrack_data[CU];
	#endif
	cl_mem K_data_buffer[CU];
	#ifdef INT_HP
		floatH* K_data[CU];
	#else
		float* K_data[CU];
	#endif

	cl_mem integrate_vol_buffer;
	cl_mem integrate_vol_sbuffer[CU];

	#ifdef FUSE_HW
	#if FUSE_NCU == 2
		cl_mem fuse_vol_sbuffer[FUSE_NCU];
	#endif
	#endif

	cl_mem pt_int_in[CU][6];
#else

	cl_kernel krnl_integrateKernel;

	cl_mem integrate_depth_buffer;
	#ifdef INT_HP
		floatH *floatDepth;
	#endif
	cl_mem volSize_buffer;
	cl_mem volDim_buffer;
	cl_mem volConst_buffer;

	cl_mem InvTrack_data_buffer;
	#ifdef INT_HP
		floatH* InvTrack_data;
	#else
		float* InvTrack_data;
	#endif
	cl_mem K_data_buffer;
	#ifdef INT_HP
		floatH* K_data;
	#else
		float* K_data;
	#endif

	cl_mem integrate_vol_buffer;
	cl_mem pt_int_in[6];

#endif

#ifdef INTKF
	#ifdef INT_NCU
		#ifdef INTKF_UNIQUEUE
			cl_mem integrate_depth_kf_buffer[CU];
			#ifdef INT_HP
				floatH *floatDepth_kf[CU];
			#else
				float *floatDepth_kf[CU];
			#endif
			cl_mem volSize_kf_buffer[CU];
			cl_mem volDim_kf_buffer[CU];
			cl_mem volConst_kf_buffer[CU];

			cl_mem InvTrack_data_kf_buffer[CU];
			#ifdef INT_HP
				floatH* InvTrack_data_kf[CU];
			#else
				float* InvTrack_data_kf[CU];
			#endif
			cl_mem K_data_kf_buffer[CU];
			#ifdef INT_HP
				floatH* K_data_kf[CU];
			#else
				float* K_data_kf[CU];
			#endif
		#endif

		cl_mem integrate_kf_vol_buffer;
		cl_mem integrate_kf_vol_sbuffer[CU];

		cl_mem pt_intkf_in[CU][6];
	#else
		#ifdef INTKF_UNIQUEUE
			cl_mem integrate_depth_kf_buffer;
			#ifdef INT_HP
				floatH *floatDepth_kf;
			#else
				float *floatDepth_kf;
			#endif
			cl_mem volSize_kf_buffer;
			cl_mem volDim_kf_buffer;
			cl_mem volConst_kf_buffer;

			cl_mem InvTrack_data_kf_buffer;
			#ifdef INT_HP
				floatH* InvTrack_data_kf;
			#else
				float* InvTrack_data_kf;
			#endif
			cl_mem K_data_kf_buffer;
			#ifdef INT_HP
				floatH* K_data_kf;
			#else
				float* K_data_kf;
			#endif
		#endif

		cl_mem integrate_kf_vol_kf_buffer;

		cl_mem pt_intkf_in[6];

	#endif
#endif

#ifdef FUSE_HW
	cl_command_queue q_fv;

	#if FUSE_NCU != 1
		cl_kernel krnl_fuseVolumesKernel[FUSE_NCU];
	#else
		cl_kernel krnl_fuseVolumesKernel;
	#endif

	cl_mem tsdf_interp_buffer[MAX_NUM_KF];
	#if FUSE_NCU != 1
		cl_mem tsdf_interp_sbuffer[MAX_NUM_KF][FUSE_NCU];
	#endif

	cl_mem volResolution_buffer;
	uint4 *volRes;

	#ifdef FUSE_ALLINONE_5
		#if FUSE_NCU != 1
			cl_mem pt_fv[FUSE_NCU][7];
		#else
			cl_mem pt_fv[7];
		#endif
	#else
		#ifdef FUSE_ALLINONE_3
			#if FUSE_NCU != 1
				cl_mem pt_fv[FUSE_NCU][5];
			#else
				cl_mem pt_fv[5];
			#endif
		#else
			cl_mem pt_fv[MAX_NUM_KF][3];
		#endif
	#endif

#endif

void initializeVitis(const char *xclbin){
	/**********************************************
	 * 
	 *          Xilinx OpenCL Initialization 
	 * 
	 * *********************************************/
	err = clGetPlatformIDs(16, platforms, &platform_count);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to find an OpenCL platform!\n");
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}
	printf("INFO: Found %d platforms\n", platform_count);
	for (cl_uint iplat=0; iplat<platform_count; iplat++) {
			err = clGetPlatformInfo(platforms[iplat], CL_PLATFORM_VENDOR, 1000, (void *)cl_platform_vendor,NULL);
		if (err != CL_SUCCESS) {
			printf("Error: clGetPlatformInfo(CL_PLATFORM_VENDOR) failed!\n");
			printf("Test failed\n");
		exit(EXIT_FAILURE);
		}
		if (strcmp(cl_platform_vendor, "Xilinx") == 0) {
			printf("INFO: Selected platform %d from %s\n", iplat, cl_platform_vendor);
			platform_id = platforms[iplat];
			platform_found = 1;
		}
	}
	if (!platform_found) {
		printf("ERROR: Platform Xilinx not found. Exit.\n");
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 16, devices, &num_devices);
	printf("INFO: Found %d devices\n", num_devices);
	if (err != CL_SUCCESS) {
		printf("ERROR: Failed to create a device group!\n");
		printf("ERROR: Test failed\n");
		exit(EXIT_FAILURE);
	}

	device_id = devices[0]; // we have only one device
	// ---------------------------------------------------------------
	// Create Context
	// ---------------------------------------------------------------
	context = clCreateContext(0,1,&device_id,NULL,NULL,&err);
	if (!context) {
		printf("Error: Failed to create a compute context!\n");
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}
	// ---------------------------------------------------------------
	// Create Command Queue
	// ---------------------------------------------------------------
	#ifdef INT_NCU
		q = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
		if (!q) {
			printf("Error: Failed to create a command q!\n");
			printf("Error: code %i\n",err);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
	#else
		q = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
		if (!q) {
			printf("Error: Failed to create a command q!\n");
			printf("Error: code %i\n",err);
			printf("Test failed\n");
			exit(EXIT_FAILURE);
		}
	#endif

	#ifdef INTKF
		#ifdef INTKF_UNIQUEUE
		#ifdef INT_NCU
			q_intkf = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
			if (!q_intkf) {
				printf("Error: Failed to create a command q_intkf!\n");
				printf("Error: code %i\n",err);
				printf("Test failed\n");
				exit(EXIT_FAILURE);
			}
		#else
			q_intkf = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
			if (!q_intkf) {
				printf("Error: Failed to create a command q_intkf!\n");
				printf("Error: code %i\n",err);
				printf("Test failed\n");
				exit(EXIT_FAILURE);
			}
		#endif
		#endif
	#endif

	#ifdef TR_HW
	q_tr = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!q_tr) {
		printf("Error: Failed to create a command q_tr!\n");
		printf("Error: code %i\n",err);
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifdef BF_HW
	q_bf = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!q_bf) {
		printf("Error: Failed to create a command q_bf!\n");
		printf("Error: code %i\n",err);
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}
	#endif

	#ifdef FUSE_HW
	q_fv = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (!q_fv) {
		printf("Error: Failed to create a command q_fv!\n");
		printf("Error: code %i\n",err);
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}
	#endif

	// ---------------------------------------------------------------
	// Load Binary File from disk
	// ---------------------------------------------------------------
	unsigned char *kernelbinary;
	cl_uint n_i0 = load_file_to_memory(xclbin, (char **) &kernelbinary);
	if (n_i0 < 0) {
		printf("failed to load kernel from xclbin: %s\n", xclbin);
		printf("Test failed\n");
		exit(EXIT_FAILURE);    
	}
	size_t n0 = n_i0;

	program = clCreateProgramWithBinary(context, 1, &device_id, &n0,
										(const unsigned char **) &kernelbinary, &status, &err);
	free(kernelbinary);
	
	if ((!program) || (err!=CL_SUCCESS)) {
		printf("Error: Failed to create compute program from binary %d!\n", err);
		exit(EXIT_FAILURE);
	}

	// -------------------------------------------------------------
	// Create Kernels
	// -------------------------------------------------------------
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(EXIT_FAILURE);
	}

	#ifdef BF_HW
		krnl_bilateralFilterKernel = clCreateKernel(program, "bilateralFilterKernel", &err);
		if (!krnl_bilateralFilterKernel || err != CL_SUCCESS) {
			printf("Error: Failed to create compute krnl_bilateralFilterKernel!\n");
			exit(EXIT_FAILURE);
		}
	#endif

	#ifdef TR_HW
		krnl_trackKernel = clCreateKernel(program, "trackKernel", &err);
		if (!krnl_trackKernel || err != CL_SUCCESS) {
			printf("Error: Failed to create compute krnl_trackKernel!\n");
			exit(EXIT_FAILURE);
		}
	#endif

	#ifdef FUSE_HW
		#if FUSE_NCU != 1
			char kernel_name1[28];

			for (int i = 0; i < FUSE_NCU; i++) {
				sprintf(kernel_name1, "fuseVolumesKernel:{f%d}", i+1);

				krnl_fuseVolumesKernel[i] = clCreateKernel(program, kernel_name1, &err);
				if (!krnl_fuseVolumesKernel[i] || err != CL_SUCCESS) {
					printf("Error: Failed to create compute krnl_fuseVolumesKernel[%d]! %d\n", i, err);
					exit(EXIT_FAILURE);
				}
			}
		#else
			krnl_fuseVolumesKernel = clCreateKernel(program, "fuseVolumesKernel", &err);
			if (!krnl_fuseVolumesKernel || err != CL_SUCCESS) {
				printf("Error: Failed to create compute krnl_fuseVolumesKernel!\n");
				exit(EXIT_FAILURE);
			}
		#endif // FUSE_NCU
	#endif

	#ifdef INT_NCU
		char kernel_name[28];

		for (int i = 0; i < CU; i++) {
			sprintf(kernel_name, "integrateKernel:{i%d}", i+1);

			krnl_integrateKernel[i] = clCreateKernel(program, kernel_name, &err);
			if (!krnl_integrateKernel[i] || err != CL_SUCCESS) {
				printf("Error: Failed to create compute krnl_integrateKernel[%d]! %d\n", i, err);
				exit(EXIT_FAILURE);
			}
		}
	#else

		krnl_integrateKernel = clCreateKernel(program, "integrateKernel", &err);
		if (!krnl_integrateKernel || err != CL_SUCCESS) {
			printf("Error: Failed to create compute krnl_integrateKernel!\n");
			exit(EXIT_FAILURE);
		}
	#endif // INT_NCU
}


/// END VITIS



void checkMemAlloc (void * ptr) {
	if (ptr == NULL) {
		std::cerr << "Error: Memory allocation problem" << std::endl;
		exit(1);
	}
}

KFusion::KFusion(Configuration config)
	:_frame(-1),
	_tracked(false),
	lastFrame(0),
	lastKeyFrame(0)
{
	if (getenv("KERNEL_TIMINGS"))
		print_kernel_timing = true;
	#ifdef KERNEL_TIMINGS
		print_kernel_timing = true;
		kernel_timings_log = fopen("kernel_timings.log","w");
	#endif

	#ifdef POSEGRAPHOPT_ENABLED
		posegraphOptEnabled = 0;
	#endif
	// kparams_t params;
	params.compute_size_ratio = config.compute_size_ratio;
	params.integration_rate = config.integration_rate;
	params.rendering_rate = config.rendering_rate;
	params.tracking_rate = config.tracking_rate;
	params.volume_resolution = config.volume_resolution;
	params.volume_direction = config.initial_pos;
	params.volume_size = config.volume_size;
	params.mu = config.mu;
	params.icp_threshold = config.icp_threshold;
	params.computationSize = make_uint2(params.inputSize.x/params.compute_size_ratio, params.inputSize.y/params.compute_size_ratio);
	params.camera = config.camera / params.compute_size_ratio;
	params.pyramid = config.pyramid;
	params.keyframe_rate = config.kf_rate;
	params.max_num_kfs = config.max_num_kfs;

	uint3 vr = make_uint3(params.volume_resolution.x,
						  params.volume_resolution.y,
						  params.volume_resolution.z);

	float3 vd = make_float3(params.volume_size.x,
							params.volume_size.y,
							params.volume_size.z);

	sMatrix4 initPose;  
	initPose(0,3) = config.initial_pos.x;
	initPose(1,3) = config.initial_pos.y;
	initPose(2,3) = config.initial_pos.z;

	pose = initPose;
	oldPose = pose;

	this->iterations.clear();
	for(auto it = params.pyramid.begin();it != params.pyramid.end(); it++)
	{    
		this->iterations.push_back(*it);
	}

	largestep = 0.75*params.mu;
	inverseCam = getInverseCameraMatrix(params.camera);
	camMatrix = getCameraMatrix(params.camera);
	step = min(params.volume_size) / max(params.volume_resolution);
	viewPose = &pose;

	uint2 cs = make_uint2(params.computationSize.x, params.computationSize.y);
	
	std::cout << "Input Size: " << params.inputSize.x << "," << params.inputSize.y << std::endl;
	std::cout << "Computation Size: " << cs.x << "," << cs.y << std::endl;

	// std::cout << "Camera" << "\n" << camMatrix << std::endl;

	// VITIS
	initializeVitis(config.binaryPath.c_str());
	std::cout << "Vitis Initialization done." << std::endl;
	
	#ifndef TR_HW
		reduction = (TrackData*)calloc(cs.x*cs.y*sizeof(TrackData), 1);
		checkMemAlloc(reduction);
		vertex = (float3*)calloc(cs.x*cs.y*sizeof(float3), 1);
		checkMemAlloc(vertex);
		normal = (float3*)calloc(cs.x*cs.y*sizeof(float3), 1);
		checkMemAlloc(normal);
	#endif

	#ifdef INT_NCU
		rawDepth = (float*)calloc(cs.x*cs.y*sizeof(float), 1);
		checkMemAlloc(rawDepth);
	#else
		#ifdef INT_HP
		rawDepth = (float*)calloc(cs.x*cs.y*sizeof(float), 1);
		checkMemAlloc(rawDepth);
		#endif
	#endif
	// integrateKernel buffers
	#ifdef INT_HP
		size_t matrix_sz = 16*sizeof(floatH);
		size_t krnl_depth_size = sizeof(floatH) * cs.x * cs.y;
	#else
		size_t matrix_sz = 16*sizeof(float);
		size_t krnl_depth_size = sizeof(float) * cs.x * cs.y;
	#endif

	
	#ifdef INT_NCU
		size_t krnl_vol_size = vr.x * vr.y *vr.z*sizeof(short2);
		
		integrate_vol_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, krnl_vol_size, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - integrate_vol_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}

		size_t sub_krnl_vol_size = vr.x * vr.y *(vr.z/CU)*sizeof(short2);
		for (int i = 0; i < CU; i++) {
			cl_buffer_region region = 	{
											i*sub_krnl_vol_size, 
											sub_krnl_vol_size
										};

			integrate_vol_sbuffer[i] = clCreateSubBuffer(integrate_vol_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateSubBuffer - integrate_vol_sbuffer[" << i << "]: " << err << std::endl;
			}
			
			integrate_depth_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,  krnl_depth_size, NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - integrate_depth_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			#ifdef INT_HP
				floatDepth[i] = (floatH *)clEnqueueMapBuffer(q,integrate_depth_buffer[i],true, CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
			#else
				floatDepth[i] = (float *)clEnqueueMapBuffer(q,integrate_depth_buffer[i],true, CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
			#endif

			volSize_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint4), NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - volSize_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			inSize[i] = (uint4 *)clEnqueueMapBuffer(q,volSize_buffer[i],true,CL_MAP_WRITE,0,sizeof(uint4),0,nullptr,nullptr,&err);

			#ifdef INT_HP
				volDim_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
			#else
				volDim_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
			#endif
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - volDim_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			#ifdef INT_HP
				inDim[i] = (floatH4 *)clEnqueueMapBuffer(q,volDim_buffer[i],true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
			#else
				inDim[i] = (float4 *)clEnqueueMapBuffer(q,volDim_buffer[i],true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
			#endif

			#ifdef INT_HP
				volConst_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
			#else
				volConst_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
			#endif
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - volConst_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			#ifdef INT_HP
				inConst[i] = (floatH4 *)clEnqueueMapBuffer(q,volConst_buffer[i],true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
			#else
				inConst[i] = (float4 *)clEnqueueMapBuffer(q,volConst_buffer[i],true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
			#endif

			InvTrack_data_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - InvTrack_data_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			#ifdef INT_HP
				InvTrack_data[i] = (floatH *)clEnqueueMapBuffer(q,InvTrack_data_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
			#else
				InvTrack_data[i] = (float *)clEnqueueMapBuffer(q,InvTrack_data_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
			#endif

			K_data_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - K_data_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			#ifdef INT_HP
				K_data[i] = (floatH *)clEnqueueMapBuffer(q,K_data_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
			#else
				K_data[i] = (float *)clEnqueueMapBuffer(q,K_data_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
			#endif
		}
		
	#else //not def INT_NCU
		size_t krnl_vol_size = vr.x * vr.y *vr.z*sizeof(short2);
		
		integrate_vol_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, krnl_vol_size, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - integrate_vol_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		integrate_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_vol_size,0,nullptr,nullptr,&err);

		integrate_depth_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,  krnl_depth_size, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - integrate_depth_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		#ifdef INT_HP
			floatDepth = (floatH *)clEnqueueMapBuffer(q,integrate_depth_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
		#else
			rawDepth = (float *)clEnqueueMapBuffer(q,integrate_depth_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
		#endif

		volSize_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint4), NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - volSize_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		inSize = (uint4 *)clEnqueueMapBuffer(q,volSize_buffer,true,CL_MAP_WRITE,0,sizeof(uint4),0,nullptr,nullptr,&err);

		#ifdef INT_HP
			volDim_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
		#else
			volDim_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
		#endif
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - volDim_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		#ifdef INT_HP
			inDim = (floatH4 *)clEnqueueMapBuffer(q,volDim_buffer,true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
		#else
			inDim = (float4 *)clEnqueueMapBuffer(q,volDim_buffer,true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
		#endif

		#ifdef INT_HP
			volConst_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
		#else
			volConst_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
		#endif
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - volConst_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		#ifdef INT_HP
			inConst = (floatH4 *)clEnqueueMapBuffer(q,volConst_buffer,true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
		#else
			inConst = (float4 *)clEnqueueMapBuffer(q,volConst_buffer,true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
		#endif

		InvTrack_data_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - InvTrack_data_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		#ifdef INT_HP
			InvTrack_data= (floatH *)clEnqueueMapBuffer(q,InvTrack_data_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
		#else
			InvTrack_data= (float *)clEnqueueMapBuffer(q,InvTrack_data_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
		#endif

		K_data_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - K_data_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		#ifdef INT_HP
			K_data = (floatH *)clEnqueueMapBuffer(q,K_data_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
		#else
			K_data = (float *)clEnqueueMapBuffer(q,K_data_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
		#endif
	#endif //INT_NCU

	#ifdef INTKF
			
		#ifdef INT_NCU
			krnl_vol_size = vr.x * vr.y *vr.z*sizeof(short2);
			
			integrate_kf_vol_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, krnl_vol_size, NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - integrate_kf_vol_buffer" << err << std::endl;
				exit(EXIT_FAILURE);
			}

			sub_krnl_vol_size = vr.x * vr.y *(vr.z/CU)*sizeof(short2);
			for (int i = 0; i < CU; i++) {
				cl_buffer_region region = 	{
												i*sub_krnl_vol_size, 
												sub_krnl_vol_size
											};
				
				integrate_kf_vol_sbuffer[i] = clCreateSubBuffer(integrate_kf_vol_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateSubBuffer - integrate_kf_vol_sbuffer[" << i << "]: " << err << std::endl;
				}
				
				#ifdef INTKF_UNIQUEUE
					integrate_depth_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,  krnl_depth_size, NULL, &err);
					if (err != CL_SUCCESS) {
						std::cout << "Return code for clCreateBuffer - integrate_depth_kf_buffer[" << i << "]: " << err << std::endl;
						exit(EXIT_FAILURE);
					}
					#ifdef INT_HP
						floatDepth_kf[i] = (floatH *)clEnqueueMapBuffer(q_intkf,integrate_depth_kf_buffer[i],true, CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
					#else
						floatDepth_kf[i] = (float *)clEnqueueMapBuffer(q_intkf,integrate_depth_kf_buffer[i],true, CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
					#endif

					volSize_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint4), NULL, &err);
					if (err != CL_SUCCESS) {
						std::cout << "Return code for clCreateBuffer - volSize_kf_buffer[" << i << "]: " << err << std::endl;
						exit(EXIT_FAILURE);
					}
					inSize_kf[i] = (uint4 *)clEnqueueMapBuffer(q_intkf,volSize_kf_buffer[i],true,CL_MAP_WRITE,0,sizeof(uint4),0,nullptr,nullptr,&err);

					#ifdef INT_HP
						volDim_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
					#else
						volDim_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
					#endif
					if (err != CL_SUCCESS) {
						std::cout << "Return code for clCreateBuffer - volDim_kf_buffer[" << i << "]: " << err << std::endl;
						exit(EXIT_FAILURE);
					}
					#ifdef INT_HP
						inDim_kf[i] = (floatH4 *)clEnqueueMapBuffer(q_intkf,volDim_kf_buffer[i],true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
					#else
						inDim_kf[i] = (float4 *)clEnqueueMapBuffer(q_intkf,volDim_kf_buffer[i],true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
					#endif

					#ifdef INT_HP
						volConst_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
					#else
						volConst_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
					#endif
					if (err != CL_SUCCESS) {
						std::cout << "Return code for clCreateBuffer - volConst_kf_buffer[" << i << "]: " << err << std::endl;
						exit(EXIT_FAILURE);
					}
					#ifdef INT_HP
						inConst_kf[i] = (floatH4 *)clEnqueueMapBuffer(q_intkf,volConst_kf_buffer[i],true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
					#else
						inConst_kf[i] = (float4 *)clEnqueueMapBuffer(q_intkf,volConst_kf_buffer[i],true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
					#endif

					InvTrack_data_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
					if (err != CL_SUCCESS) {
						std::cout << "Return code for clCreateBuffer - InvTrack_data_kf_buffer[" << i << "]: " << err << std::endl;
						exit(EXIT_FAILURE);
					}
					#ifdef INT_HP
						InvTrack_data_kf[i] = (floatH *)clEnqueueMapBuffer(q_intkf,InvTrack_data_kf_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
					#else
						InvTrack_data_kf[i] = (float *)clEnqueueMapBuffer(q_intkf,InvTrack_data_kf_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
					#endif

					K_data_kf_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
					if (err != CL_SUCCESS) {
						std::cout << "Return code for clCreateBuffer - K_data_kf_buffer[" << i << "]: " << err << std::endl;
						exit(EXIT_FAILURE);
					}
					#ifdef INT_HP
						K_data_kf[i] = (floatH *)clEnqueueMapBuffer(q_intkf,K_data_kf_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
					#else
						K_data_kf[i] = (float *)clEnqueueMapBuffer(q_intkf,K_data_kf_buffer[i],true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
					#endif
				#endif
			}
			#ifdef INTKF_UNIQUEUE
				integrate_kf_vol_ptr = (short2 *)clEnqueueMapBuffer(q_intkf,integrate_kf_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_vol_size,0,nullptr,nullptr,&err);
			#else
				integrate_kf_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_kf_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_vol_size,0,nullptr,nullptr,&err);
			#endif
			
		#else //not def INT_NCU
			krnl_vol_size = vr.x * vr.y *vr.z*sizeof(short2);
			
			integrate_kf_vol_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, krnl_vol_size, NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - integrate_kf_vol_buffer" << err << std::endl;
				exit(EXIT_FAILURE);
			}

			#ifdef INTKF_UNIQUEUE
				integrate_kf_vol_ptr = (short2 *)clEnqueueMapBuffer(q_intkf,integrate_kf_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_vol_size,0,nullptr,nullptr,&err);
			#else
				integrate_kf_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_kf_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_vol_size,0,nullptr,nullptr,&err);
			#endif

			#ifdef INTKF_UNIQUEUE
				integrate_depth_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,  krnl_depth_size, NULL, &err);
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateBuffer - integrate_depth_kf_buffer" << err << std::endl;
					exit(EXIT_FAILURE);
				}
				#ifdef INT_HP
					floatDepth_kf = (floatH *)clEnqueueMapBuffer(q_intkf,integrate_depth_kf_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
				#else
					floatDepth_kf = (float *)clEnqueueMapBuffer(q_intkf,integrate_depth_kf_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_depth_size,0,nullptr,nullptr,&err);
				#endif

				volSize_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint4), NULL, &err);
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateBuffer - volSize_kf_buffer" << err << std::endl;
					exit(EXIT_FAILURE);
				}
				inSize_kf = (uint4 *)clEnqueueMapBuffer(q_intkf,volSize_kf_buffer,true,CL_MAP_WRITE,0,sizeof(uint4),0,nullptr,nullptr,&err);

				#ifdef INT_HP
					volDim_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
				#else
					volDim_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
				#endif
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateBuffer - volDim_kf_buffer" << err << std::endl;
					exit(EXIT_FAILURE);
				}
				#ifdef INT_HP
					inDim_kf = (floatH4 *)clEnqueueMapBuffer(q_intkf,volDim_kf_buffer,true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
				#else
					inDim_kf = (float4 *)clEnqueueMapBuffer(q_intkf,volDim_kf_buffer,true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
				#endif

				#ifdef INT_HP
					volConst_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(floatH4), NULL, &err);
				#else
					volConst_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  sizeof(float4), NULL, &err);
				#endif
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateBuffer - volConst_kf_buffer" << err << std::endl;
					exit(EXIT_FAILURE);
				}
				#ifdef INT_HP
					inConst_kf = (floatH4 *)clEnqueueMapBuffer(q_intkf,volConst_kf_buffer,true,CL_MAP_WRITE,0,sizeof(floatH4),0,nullptr,nullptr,&err);
				#else
					inConst_kf = (float4 *)clEnqueueMapBuffer(q_intkf,volConst_kf_buffer,true,CL_MAP_WRITE,0,sizeof(float4),0,nullptr,nullptr,&err);
				#endif

				InvTrack_data_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateBuffer - InvTrack_data_kf_buffer" << err << std::endl;
					exit(EXIT_FAILURE);
				}
				#ifdef INT_HP
					InvTrack_data_kf = (floatH *)clEnqueueMapBuffer(q_intkf,InvTrack_data_kf_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
				#else
					InvTrack_data_kf = (float *)clEnqueueMapBuffer(q_intkf,InvTrack_data_kf_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
				#endif

				K_data_kf_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateBuffer - K_data_kf_buffer" << err << std::endl;
					exit(EXIT_FAILURE);
				}
				#ifdef INT_HP
					K_data_kf = (floatH *)clEnqueueMapBuffer(q_intkf, K_data_kf_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
				#else
					K_data_kf = (float *)clEnqueueMapBuffer(q_intkf, K_data_kf_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
				#endif
			#endif  // INTKF_UNIQUEUE
		#endif //INT_NCU
	#endif  // INTKF

	#ifdef PREF_FUSEVOL
		#ifdef FUSE_HP
			size_t size_interp_buf = vr.x * vr.y * vr.z * sizeof(floatH);
		#else
			size_t size_interp_buf = vr.x * vr.y * vr.z * sizeof(float);
		#endif

		#ifdef FUSE_HW
			for (int i = 0; i < params.max_num_kfs; i++) {
				tsdf_interp_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size_interp_buf, NULL, &err);
				if (err != CL_SUCCESS) {
					std::cout << "Return code for clCreateBuffer - tsdf_interp_buffer[" << i << "]: " << err << std::endl;
					exit(EXIT_FAILURE);
				}

				#if FUSE_NCU != 1
					size_t sub_interp_buf_size = size_interp_buf/FUSE_NCU;

					for (int k = 0; k < FUSE_NCU; k++) {
						cl_buffer_region region = 	{
														k*sub_interp_buf_size, 
														sub_interp_buf_size
													};

						tsdf_interp_sbuffer[i][k] = clCreateSubBuffer(tsdf_interp_buffer[i], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
						if (err != CL_SUCCESS) {
							std::cout << "Return code for clCreateSubBuffer - tsdf_interp_buffer[" << i << "][" << k << "]: " << err << std::endl;
						}
					}
				#endif // FUSE_NCU

				#ifdef FUSE_HW
					size_t sub_krnl_vol_size = vr.x * vr.y *(vr.z/FUSE_NCU)*sizeof(short2);

					#if FUSE_NCU == 2
						for (int i = 0; i < FUSE_NCU; i++) {
							cl_buffer_region region = 	{
															i*sub_krnl_vol_size, 
															sub_krnl_vol_size
														};

							fuse_vol_sbuffer[i] = clCreateSubBuffer(integrate_vol_buffer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
							if (err != CL_SUCCESS) {
								std::cout << "Return code for clCreateSubBuffer - fuse_vol_sbuffer[" << i << "]: " << err << std::endl;
							}
						}
					#endif
				#endif

				#ifdef FUSE_HP
					fuseVol_interp_ptr[i] = (floatH *)clEnqueueMapBuffer(q_fv,tsdf_interp_buffer[i],true, CL_MAP_WRITE,0,size_interp_buf,0,nullptr,nullptr,&err);
				#else
					fuseVol_interp_ptr[i] = (float *)clEnqueueMapBuffer(q_fv,tsdf_interp_buffer[i],true, CL_MAP_WRITE,0,size_interp_buf,0,nullptr,nullptr,&err);
				#endif
			}

			volResolution_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, sizeof(uint4), NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - volResolution_buffer" << err << std::endl;
				exit(EXIT_FAILURE);
			}
			volRes = (uint4 *)clEnqueueMapBuffer(q_fv,volResolution_buffer,true,CL_MAP_WRITE,0,sizeof(uint4),0,nullptr,nullptr,&err);

		#else
			for (int i = 0; i < params.max_num_kfs; i++) {
				fuseVol_interp_ptr[i] = (float*)calloc(size_interp_buf, 1);
				checkMemAlloc(fuseVol_interp_ptr[i]);
			}
		#endif
	#endif

	rawRgb = (uchar3*)calloc(params.inputSize.x*params.inputSize.y*sizeof(uchar3), 1);
	checkMemAlloc(rawRgb);

	scaledDepth = (float**) calloc(sizeof(float*) * iterations.size(), 1);
	checkMemAlloc(scaledDepth);

	#ifdef TR_HW
		inputVertex = (float4**) calloc(sizeof(float4*) * iterations.size(), 1);
		checkMemAlloc(inputVertex);
		inputNormal = (float4**) calloc(sizeof(float4*) * iterations.size(), 1);
		checkMemAlloc(inputNormal);
	#else
		inputVertex = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);
		checkMemAlloc(inputVertex);
		inputNormal = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);
		checkMemAlloc(inputNormal);
	#endif

	output = (float*)calloc(32*8*sizeof(float), 1);
	checkMemAlloc(output);
	
	// opencl out memory size
	#ifdef BF_HW
		krnl_out_size = sizeof(float) * (cs.x * cs.y);

		scaledDepth_zero_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY,  krnl_out_size, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - scaledDepth_zero_buffer " << err << std::endl;
			exit(EXIT_FAILURE);
		}
		scaledDepth[0] = (float *)clEnqueueMapBuffer(q_bf,scaledDepth_zero_buffer,true,CL_MAP_READ,0,krnl_out_size,0,nullptr,nullptr,&err);
	#endif

	uint lsize = cs.x * cs.y;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		#ifdef BF_HW
			if (i != 0) { // VITIS
				scaledDepth[i] = (float*) calloc(sizeof(float) * lsize, 1);
				checkMemAlloc(scaledDepth[i]);
			}
		#else
			scaledDepth[i] = (float*) calloc(sizeof(float) * lsize, 1);
			checkMemAlloc(scaledDepth[i]);
		#endif

		#ifndef TR_HW
			inputVertex[i] = (float3*) calloc(sizeof(float3) * lsize, 1);
			checkMemAlloc(inputVertex[i]);

			inputNormal[i] = (float3*) calloc(sizeof(float3) * lsize, 1);
			checkMemAlloc(inputNormal[i]);

		#else
			inVertex_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, lsize*sizeof(float4), NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - inVertex_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			inputVertex[i] = (float4*)clEnqueueMapBuffer(q_tr, inVertex_buffer[i], true, CL_MAP_READ | CL_MAP_WRITE, 0, lsize*sizeof(float4), 0, nullptr, nullptr, &err);

			inNormal_buffer[i] = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, lsize*sizeof(float4), NULL, &err);
			if (err != CL_SUCCESS) {
				std::cout << "Return code for clCreateBuffer - inNormal_buffer[" << i << "]: " << err << std::endl;
				exit(EXIT_FAILURE);
			}
			inputNormal[i] = (float4*)clEnqueueMapBuffer(q_tr, inNormal_buffer[i], true, CL_MAP_READ | CL_MAP_WRITE, 0, lsize*sizeof(float4), 0, nullptr, nullptr, &err);
		#endif
		lsize = lsize >> 2;
	}

	#ifdef TR_HW
		size_t ref_buffer_sz = sizeof(float4) * cs.x * cs.y;

		refVertex_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, ref_buffer_sz, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - refVertex_buffer: " << err << std::endl;
			exit(EXIT_FAILURE);
		}
		vertex = (float4*)clEnqueueMapBuffer(q_tr, refVertex_buffer, true, CL_MAP_READ | CL_MAP_WRITE, 0, ref_buffer_sz, 0, nullptr, nullptr, &err);

		refNormal_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, ref_buffer_sz, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - refNormal_buffer: " << err << std::endl;
			exit(EXIT_FAILURE);
		}
		normal = (float4*)clEnqueueMapBuffer(q_tr, refNormal_buffer, true, CL_MAP_READ | CL_MAP_WRITE, 0, ref_buffer_sz, 0, nullptr, nullptr, &err);

		size_t trackFloat_sz = 2*sizeof(float4)*cs.x * cs.y;
		trackData_float_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, trackFloat_sz, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - trackData_float_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		reduction = (float4 *)clEnqueueMapBuffer(q_tr,trackData_float_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,trackFloat_sz,0,nullptr,nullptr,&err);


		Ttrack_data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - Ttrack_data_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		Ttrack_data = (float *)clEnqueueMapBuffer(q_tr,Ttrack_data_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);

		view_data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_sz, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - view_data_buffer" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		view_data = (float *)clEnqueueMapBuffer(q_tr,view_data_buffer,true,CL_MAP_WRITE,0,matrix_sz,0,nullptr,nullptr,&err);
	#endif

	// bilateralFilter kernel buffers
	#ifdef BF_HW

		padSize.x = cs.x + 2;
		padSize.y = cs.y + 2;

		krnl_paddepth_size = sizeof(float)*padSize.x*padSize.y;
		floatDepth_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, krnl_paddepth_size, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - pad_depth" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		pad_depth = (float *)clEnqueueMapBuffer(q_bf, floatDepth_buffer, true, CL_MAP_WRITE, 0, krnl_paddepth_size, 0, nullptr, nullptr, &err);

		size_t gaussianS = 9;
		krnl_gaussian_size = gaussianS * sizeof(float);

		gaussian_buffer = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY,  krnl_gaussian_size, NULL, &err);
		if (err != CL_SUCCESS) {
			std::cout << "Return code for clCreateBuffer - gaussian" << err << std::endl;
			exit(EXIT_FAILURE);
		}
		gaussian = (float *)clEnqueueMapBuffer(q_bf, gaussian_buffer, true, CL_MAP_WRITE, 0, krnl_gaussian_size, 0, nullptr, nullptr, &err);
		
		if (!(floatDepth_buffer&&gaussian_buffer&&scaledDepth_zero_buffer)) {
			printf("Error: Failed to allocate device memory!\n");
			exit(EXIT_FAILURE);
		}

		pt_bf[0] = floatDepth_buffer;
		pt_bf[1] = gaussian_buffer;
		pt_bf[2] = scaledDepth_zero_buffer;
	#else
		// Generate the gaussian
		size_t gaussianS = 2*radius + 1;
		gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
		checkMemAlloc(gaussian);

		int x;
		for (unsigned int i = 0; i < 5; i++) {
			x = i - 2;
			gaussian[i] = expf(-(x * x) / (2 * delta * delta));
		}
	#endif

	#ifdef BF_HW
		gaussian[0] = 0.9394130111;
		gaussian[1] = 0.9692332149;
		gaussian[2] = 0.9394130111;
		gaussian[3] = 0.9692332149;
		gaussian[4] = 1;
		gaussian[5] = 0.9692332149;
		gaussian[6] = 0.9394130111;
		gaussian[7] = 0.9692332149;
		gaussian[8] = 0.9394130111;
	#endif

	#ifdef INT_NCU
		for (int i = 0; i < CU; i++) {
			inSize[i]->x = vr.x;
			inSize[i]->y = vr.y;
			inSize[i]->z = vr.z;

			inDim[i]->x = vd.x;
			inDim[i]->y = vd.y;
			inDim[i]->z = vd.z;

			inConst[i]->x = vd.x / vr.x;
			inConst[i]->y = vd.y / vr.y;
			inConst[i]->z = vd.z / vr.z;
		}
	#else
		inSize->x = vr.x;
		inSize->y = vr.y;
		inSize->z = vr.z;

		inDim->x = vd.x;
		inDim->y = vd.y;
		inDim->z = vd.z;

		inConst->x = vd.x / vr.x;
		inConst->y = vd.y / vr.y;
		inConst->z = vd.z / vr.z;
	#endif // INT_NCU

	#ifdef INTKF
		#ifdef INTKF_UNIQUEUE
			#ifdef INT_NCU
				for (int i = 0; i < CU; i++) {
					inSize_kf[i]->x = vr.x;
					inSize_kf[i]->y = vr.y;
					inSize_kf[i]->z = vr.z;

					inDim_kf[i]->x = vd.x;
					inDim_kf[i]->y = vd.y;
					inDim_kf[i]->z = vd.z;

					inConst_kf[i]->x = vd.x / vr.x;
					inConst_kf[i]->y = vd.y / vr.y;
					inConst_kf[i]->z = vd.z / vr.z;
				}
			#else
				inSize_kf->x = vr.x;
				inSize_kf->y = vr.y;
				inSize_kf->z = vr.z;

				inDim_kf->x = vd.x;
				inDim_kf->y = vd.y;
				inDim_kf->z = vd.z;

				inConst_kf->x = vd.x / vr.x;
				inConst_kf->y = vd.y / vr.y;
				inConst_kf->z = vd.z / vr.z;
			#endif // INT_NCU
		#endif
	#endif

	#ifdef FUSE_HW
		volRes->x = vr.x;
		volRes->y = vr.y;
		volRes->z = vr.z;
	#endif

	integrate_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,krnl_vol_size,0,nullptr,nullptr,&err);
	volume.init(vr, vd, integrate_vol_ptr);

	#ifdef INTKF
		keyFrameVol.init(vr, vd, integrate_kf_vol_ptr);
	#else
		keyFrameVol.init(vr, vd, NULL);
	#endif
	fusionVol.init(vr, vd, NULL);

	initVolumeKernel(volume);
	
	err = clEnqueueUnmapMemObject(q,integrate_vol_buffer,integrate_vol_ptr,0,NULL,NULL);
	if(err != CL_SUCCESS){
		printf("Error: Failed to unmap device memory integrate_vol_buffer!\n");
	}
	clFinish(q);
	initVolumeKernel(fusionVol);
	initVolumeKernel(keyFrameVol);
	
}

void KFusion::vitisRelease() {
	// Cleanup OpenCL objects
	// bilateralFilter kernel
	#ifdef BF_HW
		err = clEnqueueUnmapMemObject(q_bf,floatDepth_buffer,pad_depth,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory floatDepth_buffer!\n");
		}
		err = clEnqueueUnmapMemObject(q_bf, gaussian_buffer, gaussian,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory floatDepth_buffer!\n");
		}
		err = clEnqueueUnmapMemObject(q_bf,scaledDepth_zero_buffer,scaledDepth[0],0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory floatDepth_buffer!\n");
		}
		clReleaseMemObject(floatDepth_buffer);
		clReleaseMemObject(gaussian_buffer);
		clReleaseMemObject(scaledDepth_zero_buffer);
	#endif

	// // trackKernel
	#ifdef TR_HW
		for (int i = 0; i < iterations.size(); i++) {
			err = clEnqueueUnmapMemObject(q_tr,inVertex_buffer[i],inputVertex[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory inVertex_buffer[%d]!\n", i);
			}

			err = clEnqueueUnmapMemObject(q_tr,inNormal_buffer[i],inputNormal[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory inNormal_buffer[%d]!\n", i);
			}
		}

		err = clEnqueueUnmapMemObject(q_tr,refVertex_buffer,vertex,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory refVertex_buffer!\n");
		}

		err = clEnqueueUnmapMemObject(q_tr,refNormal_buffer,normal,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory refNormal_buffer!\n");
		}

		err = clEnqueueUnmapMemObject(q_tr,trackData_float_buffer,reduction,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory trackData_float_buffer!\n");
		}

		err = clEnqueueUnmapMemObject(q_tr,Ttrack_data_buffer,Ttrack_data,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory Ttrack_data_buffer!\n");
		}

		err = clEnqueueUnmapMemObject(q_tr,view_data_buffer,view_data,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory view_data_buffer!\n");
		}
		clFinish(q_tr);

		clReleaseMemObject(refVertex_buffer);
		clReleaseMemObject(refNormal_buffer);
		clReleaseMemObject(trackData_float_buffer);
		clReleaseMemObject(Ttrack_data_buffer);
		clReleaseMemObject(view_data_buffer);
	#endif

	// integrateKernel
	#ifdef INT_NCU

		for (int i = 0; i < CU; i++) {
			err = clEnqueueUnmapMemObject(q,integrate_depth_buffer[i],floatDepth[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory floatDepth_buffer[%d]!\n", i);
			}
			err = clEnqueueUnmapMemObject(q,volSize_buffer[i],inSize[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory volSize_buffer[%d]!\n", i);
			}
			err = clEnqueueUnmapMemObject(q,volDim_buffer[i],inDim[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory volDim_buffer[%d]!\n", i);
			}
			err = clEnqueueUnmapMemObject(q,volConst_buffer[i],inConst[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory volConst_buffer[%d]!\n", i);
			}
			err = clEnqueueUnmapMemObject(q,InvTrack_data_buffer[i],InvTrack_data[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory InvTrack_data[%d]!\n", i);
			}
			err = clEnqueueUnmapMemObject(q,K_data_buffer[i],K_data[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory K_data[%d]!\n", i);
			}

			clFinish(q);

			err = clReleaseMemObject(volSize_buffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release volSize_buffer[%d]\n", i);
			err = clReleaseMemObject(volDim_buffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release volDim_buffer[%d]\n", i);
			err = clReleaseMemObject(volConst_buffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release volConst_buffer[%d]\n", i);
			err = clReleaseMemObject(InvTrack_data_buffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release InvTrack_data_buffer[%d]\n", i);
			err = clReleaseMemObject(K_data_buffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release K_data_buffer[%d]\n", i);

			err = clReleaseMemObject(integrate_depth_buffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release integrate_depth_buffer[%d]\n", i);
			err = clReleaseMemObject(integrate_vol_sbuffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release integrate_vol_sbuffer[%d]\n", i);
		}

		err = clReleaseMemObject(integrate_vol_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release \n");
		
	#else // NOT INT_NCU
		err = clEnqueueUnmapMemObject(q,integrate_vol_buffer,integrate_vol_ptr,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory integrate_vol_buffer!\n");
		}

		#ifdef INT_HP
			err = clEnqueueUnmapMemObject(q,integrate_depth_buffer,floatDepth,0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory floatDepth_buffer!\n");
			}
		#else
			err = clEnqueueUnmapMemObject(q,integrate_depth_buffer,rawDepth,0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory floatDepth_buffer!\n");
			}
		#endif

		err = clEnqueueUnmapMemObject(q,volSize_buffer,inSize,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory volSize_buffer!\n");
		}
		err = clEnqueueUnmapMemObject(q,volDim_buffer,inDim,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory volDim_buffer!\n");
		}
		err = clEnqueueUnmapMemObject(q,volConst_buffer,inConst,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory volConst_buffer!\n");
		}
		err = clEnqueueUnmapMemObject(q,InvTrack_data_buffer,InvTrack_data,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory InvTrack_data!\n");
		}
		err = clEnqueueUnmapMemObject(q,K_data_buffer,K_data,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory K_data!\n");
		}

		clFinish(q);

		err = clReleaseMemObject(volSize_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release volSize_buffer: %d\n", err);
		err = clReleaseMemObject(volDim_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release volDim_buffer: %d\n", err);
		err = clReleaseMemObject(volConst_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release volConst_buffer: %d\n", err);
		err = clReleaseMemObject(InvTrack_data_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release InvTrack_data_buffer: %d\n", err);
		err = clReleaseMemObject(K_data_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release K_data_buffer: %d\n", err);

		err = clReleaseMemObject(integrate_depth_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release K_data_buffer: %d\n", err);
		err = clReleaseMemObject(integrate_vol_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release integrate_vol_buffer: %d\n", err);

	#endif // INT_NCU

	clFinish(q);

	#ifdef INTKF
		#ifdef INT_NCU
			#ifndef INTKF_UNIQUEUE
				err = clEnqueueUnmapMemObject(q,integrate_kf_vol_buffer,integrate_kf_vol_ptr,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory integrate_kf_vol_buffer!\n");
				}
				clFinish(q);
			#else
				err = clEnqueueUnmapMemObject(q_intkf,integrate_kf_vol_buffer,integrate_kf_vol_ptr,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory integrate_kf_vol_buffer!\n");
				}
				clFinish(q_intkf);


				for (int i = 0; i < CU; i++) {
					err = clEnqueueUnmapMemObject(q_intkf,integrate_depth_kf_buffer[i],floatDepth_kf[i],0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory floatDepth_kf_buffer[%d]!\n", i);
					}
					err = clEnqueueUnmapMemObject(q_intkf,volSize_kf_buffer[i],inSize_kf[i],0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory volSize_kf_buffer[%d]!\n", i);
					}
					err = clEnqueueUnmapMemObject(q_intkf,volDim_kf_buffer[i],inDim_kf[i],0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory volDim_kf_buffer[%d]!\n", i);
					}
					err = clEnqueueUnmapMemObject(q_intkf,volConst_kf_buffer[i],inDim_kf[i],0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory volConst_kf_buffer[%d]!\n", i);
					}
					err = clEnqueueUnmapMemObject(q_intkf,InvTrack_data_kf_buffer[i],InvTrack_data_kf[i],0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory InvTrack_data_kf[%d]!\n", i);
					}
					err = clEnqueueUnmapMemObject(q_intkf,K_data_kf_buffer[i],K_data_kf[i],0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory K_data_kf[%d]!\n", i);
					}

					clFinish(q_intkf);

					err = clReleaseMemObject(volSize_kf_buffer[i]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release volSize_kf_buffer[%d]\n", i);
					err = clReleaseMemObject(volDim_kf_buffer[i]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release volDim_kf_buffer[%d]\n", i);
					err = clReleaseMemObject(volConst_kf_buffer[i]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release volConst_kf_buffer[%d]\n", i);
					err = clReleaseMemObject(InvTrack_data_kf_buffer[i]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release InvTrack_data_kf_buffer[%d]\n", i);
					err = clReleaseMemObject(K_data_kf_buffer[i]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release K_data_kf_buffer[%d]\n", i);

					err = clReleaseMemObject(integrate_depth_kf_buffer[i]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release integrate_depth_kf_buffer[%d]\n", i);
					
					err = clReleaseMemObject(integrate_kf_vol_sbuffer[i]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release integrate_kf_vol_sbuffer[%d]\n", i);
				}
			#endif

			err = clReleaseMemObject(integrate_kf_vol_buffer);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release integrate_kf_vol_buffer\n");
			
		#else // NOT INT_NCU
			#ifdef INTKF_UNIQUEUE
				#ifdef INT_HP
					err = clEnqueueUnmapMemObject(q_intkf,integrate_depth_kf_buffer,floatDepth_kf,0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory floatDepth_kf_buffer!\n");
					}
				#else
					err = clEnqueueUnmapMemObject(q_intkf,integrate_depth_kf_buffer,floatDepth_kf,0,NULL,NULL);
					if(err != CL_SUCCESS){
						printf("Error: Failed to unmap device memory floatDepth_kf_buffer!\n");
					}
				#endif

				err = clEnqueueUnmapMemObject(q_intkf,volSize_buffer_kf,inSize_kf,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory volSize_kf_buffer!\n");
				}
				err = clEnqueueUnmapMemObject(q_intkf,volDim_kf_buffer,inDim_kf,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory volDim_kf_buffer!\n");
				}
				err = clEnqueueUnmapMemObject(q_intkf,volConst_kf_buffer,inDim_kf,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory volConst_kf_buffer!\n");
				}
				err = clEnqueueUnmapMemObject(q_intkf,InvTrack_data_kf_buffer,InvTrack_data_kf,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory InvTrack_data_kf!\n");
				}
				err = clEnqueueUnmapMemObject(q_intkf,K_data_kf_buffer,K_data_kf,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory K_data_kf!\n");
				}

				clFinish(q_intkf);

				err = clEnqueueUnmapMemObject(q_intkf,integrate_kf_vol_buffer,integrate_kf_vol_ptr,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory integrate_kf_vol_buffer!\n");
				}

				clFinish(q_intkf);

				err = clReleaseMemObject(volSize_kf_buffer);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release volSize_kf_buffer: %d\n", err);
				err = clReleaseMemObject(volDim_kf_buffer);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release volDim_kf_buffer: %d\n", err);
				err = clReleaseMemObject(volConst_kf_buffer);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release volConst_kf_buffer: %d\n", err);
				err = clReleaseMemObject(InvTrack_data_kf_buffer);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release InvTrack_data_kf_buffer: %d\n", err);
				err = clReleaseMemObject(K_data_kf_buffer);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release K_data_kf_buffer: %d\n", err);

				err = clReleaseMemObject(integrate_depth_kf_buffer);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release integrate_depth_kf_buffer: %d\n", err);
				err = clReleaseMemObject(integrate_kf_vol_buffer);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release integrate_kf_vol_buffer: %d\n", err);
			#else
				err = clEnqueueUnmapMemObject(q,integrate_kf_vol_buffer,integrate_kf_vol_ptr,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory integrate_kf_vol_buffer!\n");
				}

				clFinish(q);
			#endif

		#endif // INT_NCU
	#endif

	#ifdef FUSE_HW
		err = clEnqueueUnmapMemObject(q_fv,volResolution_buffer,volRes,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory volResolution_buffer!\n");
		}

		for (int i = 0; i < params.max_num_kfs; i++) {
			err = clEnqueueUnmapMemObject(q_fv,tsdf_interp_buffer[i],fuseVol_interp_ptr[i],0,NULL,NULL);
			if(err != CL_SUCCESS){
				printf("Error: Failed to unmap device memory tsdf_interp_buffer[%d]!\n", i);
			}

			clFinish(q_fv);

			#if FUSE_NCU != 1
				for (int k = 0; k < FUSE_NCU; k++) {
					err = clReleaseMemObject(tsdf_interp_sbuffer[i][k]);
					if (err != CL_SUCCESS)
						printf("Error: Failed to release tsdf_interp_sbuffer[%d][%d]: %d\n", i, k, err);
				}
			#endif

			err = clReleaseMemObject(tsdf_interp_buffer[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release tsdf_interp_buffer[%d]: %d\n", i, err);
		}

		err = clReleaseMemObject(volResolution_buffer);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release volResolution_buffer: %d\n", err);
		
	#endif

	err = clReleaseProgram(program);
	if (err != CL_SUCCESS)
		printf("Error: Failed to release program: %d\n", err);

	#ifdef INT_NCU
		for (int i = 0; i < CU; i++) {
			err = clReleaseKernel(krnl_integrateKernel[i]);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release krnl_integrateKernel[%d]: %d\n", i, err);
		}
	#else
		err = clReleaseKernel(krnl_integrateKernel);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release Int kernel: %d\n", err);
	#endif

	#ifdef BF_HW
		err = clReleaseKernel(krnl_bilateralFilterKernel);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release BF kernel: %d\n", err);
	#endif

	#ifdef TR_HW
		err = clReleaseKernel(krnl_trackKernel);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release Tr kernel: %d\n", err);

		err = clReleaseCommandQueue(q_tr);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release command queue (q_tr): %d\n", err);
	#endif

	#ifdef FUSE_HW
		#if FUSE_NCU != 1
			for (int i = 0; i < FUSE_NCU; i++) {
				err = clReleaseKernel(krnl_fuseVolumesKernel[i]);
				if (err != CL_SUCCESS)
					printf("Error: Failed to release fuseVolumes kernel[%d]: %d\n", i, err);
			}
		#else
			err = clReleaseKernel(krnl_fuseVolumesKernel);
			if (err != CL_SUCCESS)
				printf("Error: Failed to release fuseVolumes kernel: %d\n", err);
		#endif

		err = clReleaseCommandQueue(q_fv);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release command queue (q_fv): %d\n", err);
	#endif

	err = clReleaseCommandQueue(q);
	if (err != CL_SUCCESS)
		printf("Error: Failed to release command queue: %d\n", err);

	#ifdef INTKF
	#ifdef INTKF_UNIQUEUE
		err = clReleaseCommandQueue(q_intkf);
		if (err != CL_SUCCESS)
			printf("Error: Failed to release command queue (q_intkf): %d\n", err);
	#endif
	#endif

	err = clReleaseContext(context);
	if (err != CL_SUCCESS)
		printf("Error: Failed to release context: %d\n", err);

}


KFusion::~KFusion()
{
	#ifdef KERNEL_TIMINGS
		fclose(kernel_timings_log);
	#endif

	vitisRelease();
	std::cout << "Vitis release done" << std::endl;

	fusionVol.release();
	#ifndef INTKF
		keyFrameVol.release();
	#endif
	
	#ifndef TR_HW
		free(reduction);
		free(normal);
		free(vertex);
	#endif
	
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		#ifdef BF_HW
			if (i != 0)    // VITIS
				free(scaledDepth[i]);
		#else
			free(scaledDepth[i]);
		#endif

		#ifndef TR_HW
			free(inputVertex[i]);
			free(inputNormal[i]);
		#endif
	}

	free(scaledDepth);
	free(inputVertex);
	free(inputNormal);
	
	#ifdef INT_NCU
		free(rawDepth);
	#else
	#ifdef INT_HP
		free(rawDepth);
	#endif
	#endif
	
	free(rawRgb);
	free(output);
	#ifndef BF_HW
		free(gaussian);
	#endif
	
}


void KFusion::dropKeyFrame(int val)
{
	for(auto it = volumes.begin(); it != volumes.end(); it++) 
	{
		if(it->frame == val)
		{
			short2 *data=it->data;
			volumes.erase(it); 
			#ifndef CHANGE_VOLS
				delete[] data;
			#endif
			return;
		}
	}
}

void initVolumeKernel(Volume volume) {
	TICK();
	unsigned int z;

	#pragma omp parallel for private(z)
	for (z = 0; z < volume._resolution.z; z++) {
		for (unsigned int x = 0; x < volume._resolution.x; x++) {
			for (unsigned int y = 0; y < volume._resolution.y; y++) {
				volume.setints(x, y, z, make_float2(1.0f, 0.0f));
			}
		}
	}
	TOCK("initVolumeKernel", volume._resolution.x*volume._resolution.y*volume._resolution.z);
}


#ifdef BF_HW
	void bilateralFilterKernel(float* out, float* in, uint size_x, uint size_y,
			const float * gaussian, float e_d, int r) {

		TICK();
		int start;
		int end;

		start = 0;
		end = 240;

		int argcounter = 0;
		// Set arguments 
		err = 0;
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(cl_mem), &scaledDepth_zero_buffer);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(cl_mem), &floatDepth_buffer);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(uint), &size_x);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(uint), &size_y);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(cl_mem), &gaussian_buffer);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(float), &e_d);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(int), &r);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(int), &start);
		err |= clSetKernelArg(krnl_bilateralFilterKernel, argcounter++, sizeof(int), &end);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to set kernel_vector_add arguments! %d\n", err);
	 
		}
		err = clEnqueueMigrateMemObjects(q_bf, (cl_uint)2, pt_bf, 0 ,0,NULL, NULL);

		
		err = clEnqueueTask(q_bf, krnl_bilateralFilterKernel, 0, NULL, NULL);
		if (err) {
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(EXIT_FAILURE);
		}
		err = 0;
		err |= clEnqueueMigrateMemObjects(q_bf, (cl_uint)1, &pt_bf[2], CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to write to source array: %d!\n", err);
			exit(EXIT_FAILURE);
		}

		clFinish(q_bf);
		TOCK("bilateralFilterKernel", padSize.x*padSize.y);
	}
#else

	void bilateralFilterKernel(float* out, const float* in, uint2 size,
			const float * gaussian, float e_d, int r) {
		TICK()
		uint y;
		float e_d_squared_2 = e_d * e_d * 2;
		
		#pragma omp parallel for \
			shared(out),private(y)   
		for (y = 0; y < size.y; y++) {
			for (uint x = 0; x < size.x; x++) {
				uint pos = x + y * size.x;
				if (in[pos] == 0) {
					out[pos] = 0;
					continue;
				}

				float sum = 0.0f;
				float t = 0.0f;

				const float center = in[pos];

				for (int i = -r; i <= r; ++i) {
					for (int j = -r; j <= r; ++j) {
						uint2 curPos = make_uint2(clamp(x + i, 0u, size.x - 1),
								clamp(y + j, 0u, size.y - 1));
						const float curPix = in[curPos.x + curPos.y * size.x];

						if (curPix > 0) {
							const float mod = (curPix - center)*(curPix - center);
							const float factor = gaussian[i + r]
									* gaussian[j + r]
									* expf(-mod / e_d_squared_2);
							t += factor * curPix;
							sum += factor;
						}
					}
				}
				out[pos] = t / sum;
			}
		}

		TOCK("bilateralFilterKernel", size.x*size.y);
	}
#endif


void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
		const float e_d, const int r) {
	TICK();
	uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
	unsigned int y;

	#pragma omp parallel for \
			shared(out), private(y)
	for (y = 0; y < outSize.y; y++) {
		for (unsigned int x = 0; x < outSize.x; x++) {
			uint2 pixel = make_uint2(x, y);
			const uint2 centerPixel = 2 * pixel;

			float sum = 0.0f;
			float t = 0.0f;
			float center = in[centerPixel.x
					+ centerPixel.y * inSize.x];

			for (int i = -r + 1; i <= r; ++i) {
				for (int j = -r + 1; j <= r; ++j) {
					uint2 cur = make_uint2(
							clamp(
									make_int2(centerPixel.x + j,
											centerPixel.y + i), make_int2(0),
									make_int2(2 * outSize.x - 1,
											2 * outSize.y - 1)));
					float current = in[cur.x + cur.y * inSize.x];
					if (fabsf(current - center) < e_d) {
						sum += 1.0f;
						t += current;
					}
				}
			}
			out[pixel.x + pixel.y * outSize.x] = t / sum;
		}
	}
	TOCK("halfSampleRobustImageKernel", outSize.x*outSize.y);
}

#ifdef TR_HW
void depth2vertexKernel(float4* vertex, const float * depth, uint2 imageSize,
		const sMatrix4 invK) {
	TICK();
	unsigned int x, y;

	#pragma omp parallel for \
			 shared(vertex), private(x, y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {

			if (depth[x + y * imageSize.x] > 0) {
				vertex[x + y * imageSize.x] = make_float4(depth[x + y * imageSize.x]
						* (rotate(invK, make_float3(x, y, 1.f))));
			} else {
				vertex[x + y * imageSize.x] = make_float4(0);
			}
		}
	}
	TOCK("depth2vertexKernel", imageSize.x*imageSize.y);
}

void vertex2normalKernel(float4 * out, const float4 * in, uint2 imageSize) {
	TICK();
	unsigned int x, y;
	
	#pragma omp parallel for \
			shared(out), private(x,y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {
			const uint2 pleft = make_uint2(max(int(x) - 1, 0), y);
			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1),
					y);
			const uint2 pup = make_uint2(x, max(int(y) - 1, 0));
			const uint2 pdown = make_uint2(x,
					min(y + 1, ((int) imageSize.y) - 1));

			const float4 left = in[pleft.x + imageSize.x * pleft.y];
			const float4 right = in[pright.x + imageSize.x * pright.y];
			const float4 up = in[pup.x + imageSize.x * pup.y];
			const float4 down = in[pdown.x + imageSize.x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y * imageSize.x].x = KFUSION_INVALID;
				continue;
			}
			const float3 dxv = make_float3(right - left);
			const float3 dyv = make_float3(down - up);
			out[x + y * imageSize.x] = make_float4(normalize(cross(dyv, dxv))); // switched dx and dy to get factor -1
		}
	}
	TOCK("vertex2normalKernel", imageSize.x*imageSize.y);
}

#else
#ifdef TR_SW_APPROX
void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize,
		const sMatrix4 invK) {
	TICK();
	unsigned int x, y;

	#pragma omp parallel for \
			 shared(vertex), private(x, y)
	for (y = 0; y < imageSize.y; y+=4) {
		for (x = 0; x < imageSize.x; x+=8) {

			if (depth[x + y * imageSize.x] > 0) {
				vertex[x + y * imageSize.x] = depth[x + y * imageSize.x]
						* (rotate(invK, make_float3(x, y, 1.f)));
			} else {
				vertex[x + y * imageSize.x] = make_float3(0);
			}

			if (x > 0) {
				if (depth[x - 1 + y * imageSize.x] > 0) {
					vertex[x - 1 + y * imageSize.x] = depth[x - 1 + y * imageSize.x]
							* (rotate(invK, make_float3(x-1, y, 1.f)));
				} else {
					vertex[x - 1 + y * imageSize.x] = make_float3(0);
				}
			}

			if (x < imageSize.x - 1) {
				if (depth[x + 1 + y * imageSize.x] > 0) {
					vertex[x + 1 + y * imageSize.x] = depth[x + 1 + y * imageSize.x]
							* (rotate(invK, make_float3(x+1, y, 1.f)));
				} else {
					vertex[x + 1 + y * imageSize.x] = make_float3(0);
				}
			}

			if (y > 0) {
				if (depth[x + (y - 1) * imageSize.x] > 0) {
					vertex[x + (y-1) * imageSize.x] = depth[x + (y-1) * imageSize.x]
							* (rotate(invK, make_float3(x, y-1, 1.f)));
				} else {
					vertex[x + (y-1) * imageSize.x] = make_float3(0);
				}
			}

			if (y < imageSize.y - 1) {
				if (depth[x + (y+1) * imageSize.x] > 0) {
					vertex[x + (y+1) * imageSize.x] = depth[x + (y+1) * imageSize.x]
							* (rotate(invK, make_float3(x, y+1, 1.f)));
				} else {
					vertex[x + (y + 1) * imageSize.x] = make_float3(0);
				}
			}

		}
	}
	TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize) {
	TICK();
	unsigned int x, y;

	#pragma omp parallel for \
			shared(out), private(x,y)
	for (y = 0; y < imageSize.y; y+=4) {
		for (x = 0; x < imageSize.x; x+=8) {
			const uint2 pleft = make_uint2(max(int(x) - 1, 0), y);
			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1),
					y);
			const uint2 pup = make_uint2(x, max(int(y) - 1, 0));
			const uint2 pdown = make_uint2(x,
					min(y + 1, ((int) imageSize.y) - 1));

			const float3 left = in[pleft.x + imageSize.x * pleft.y];
			const float3 right = in[pright.x + imageSize.x * pright.y];
			const float3 up = in[pup.x + imageSize.x * pup.y];
			const float3 down = in[pdown.x + imageSize.x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y * imageSize.x].x = KFUSION_INVALID;
				continue;
			}
			const float3 dxv = right - left;
			const float3 dyv = down - up;
			out[x + y * imageSize.x] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
		}
	}
	TOCK("vertex2normalKernel", imageSize.x * imageSize.y);
}

#else
void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize,
		const sMatrix4 invK) {
	TICK();
	unsigned int x, y;

	#pragma omp parallel for \
			 shared(vertex), private(x, y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {

			if (depth[x + y * imageSize.x] > 0) {
				vertex[x + y * imageSize.x] = depth[x + y * imageSize.x]
						* (rotate(invK, make_float3(x, y, 1.f)));
			} else {
				vertex[x + y * imageSize.x] = make_float3(0);
			}
		}
	}
	TOCK("depth2vertexKernel", imageSize.x*imageSize.y);
}

void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize) {
	TICK();
	unsigned int x, y;

	#pragma omp parallel for \
			shared(out), private(x,y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {
			const uint2 pleft = make_uint2(max(int(x) - 1, 0), y);
			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1),
					y);
			const uint2 pup = make_uint2(x, max(int(y) - 1, 0));
			const uint2 pdown = make_uint2(x,
					min(y + 1, ((int) imageSize.y) - 1));

			const float3 left = in[pleft.x + imageSize.x * pleft.y];
			const float3 right = in[pright.x + imageSize.x * pright.y];
			const float3 up = in[pup.x + imageSize.x * pup.y];
			const float3 down = in[pdown.x + imageSize.x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y * imageSize.x].x = KFUSION_INVALID;
				continue;
			}
			const float3 dxv = right - left;
			const float3 dyv = down - up;
			out[x + y * imageSize.x] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
		}
	}
	TOCK("vertex2normalKernel", imageSize.x*imageSize.y);
}
#endif
#endif


#ifdef TR_HW
void reduceKernel(float * out, float4* J, const uint2 Jsize,
		const uint2 size) {
	TICK();
	int blockIndex;
	
	#pragma omp parallel for private (blockIndex)
	for (blockIndex = 0; blockIndex < 8; blockIndex++) {

		float S[112][32]; // this is for the final accumulation
		// we have 112 threads in a blockdim
		// and 8 blocks in a gridDim?
		// ie it was launched as <<<8,112>>>
		
		for(int threadIndex = 0; threadIndex < 112; threadIndex++) {
			uint sline = threadIndex;
			float sums[32];
			float * jtj = sums+7;
			float * info = sums+28;
			for(uint i = 0; i < 32; ++i) sums[i] = 0;

			for(uint y = blockIndex; y < size.y; y += 8) {
				for(uint x = sline; x < size.x; x += 112) {
					int idx = x + y * Jsize.x;
					const float4 row1 = J[2*idx];
					const float4 row2 = J[2*idx + 1];

					if(row1.x < 1) {
						// accesses S[threadIndex][28..31]
						info[1] += row1.x == -4 ? 1 : 0;
						info[2] += row1.x == -5 ? 1 : 0;
						info[3] += row1.x > -4 ? 1 : 0;
						continue;
					}
					// Error part
					sums[0] += row1.y * row1.y;

					// JTe part
					sums[1] += row1.y * row1.z;
					sums[2] += row1.y * row1.w;
					sums[3] += row1.y * row2.x;
					sums[4] += row1.y * row2.y;
					sums[5] += row1.y * row2.z;
					sums[6] += row1.y * row2.w;

					// JTJ part, unfortunatly the double loop is not unrolled well...
					jtj[0] += row1.z * row1.z;
					jtj[1] += row1.z * row1.w;
					jtj[2] += row1.z * row2.x;
					jtj[3] += row1.z * row2.y;

					jtj[4] += row1.z * row2.z;
					jtj[5] += row1.z * row2.w;

					jtj[6] += row1.w * row1.w;
					jtj[7] += row1.w * row2.x;
					jtj[8] += row1.w * row2.y;
					jtj[9] += row1.w * row2.z;

					jtj[10] += row1.w * row2.w;

					jtj[11] += row2.x * row2.x;
					jtj[12] += row2.x * row2.y;
					jtj[13] += row2.x * row2.z;
					jtj[14] += row2.x * row2.w;

					jtj[15] += row2.y * row2.y;
					jtj[16] += row2.y * row2.z;
					jtj[17] += row2.y * row2.w;

					jtj[18] += row2.z * row2.z;
					jtj[19] += row2.z * row2.w;

					jtj[20] += row2.w * row2.w;

					// extra info here
					info[0] += 1;
				}
			}

			for(int i = 0; i < 32; ++i) { // copy over to shared memory
				S[sline][i] = sums[i];
			}
			// WE NO LONGER NEED TO DO THIS AS the threads execute sequentially inside a for loop

		}

		for(int ssline = 0; ssline < 32; ssline++) { // sum up columns and copy to global memory in the final 32 threads
			for(unsigned i = 1; i < 112; ++i) {
				S[0][ssline] += S[i][ssline];
			}
			out[ssline+blockIndex*32] = S[0][ssline];
		}
	}

	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
	for (int j = 1; j < 8; ++j) {
		values[0] += values[j];
	}

	TOCK("reduceKernel", 512);
}

#else

void reduceKernel(float * out, TrackData* J, const uint2 Jsize,
		const uint2 size) {
	TICK();
	int blockIndex;
	
	#pragma omp parallel for private (blockIndex)
	for (blockIndex = 0; blockIndex < 8; blockIndex++) {

		float S[112][32]; // this is for the final accumulation
		// we have 112 threads in a blockdim
		// and 8 blocks in a gridDim?
		// ie it was launched as <<<8,112>>>
		
		for(int threadIndex = 0; threadIndex < 112; threadIndex++) {
			uint sline = threadIndex;
			float sums[32];
			float * jtj = sums+7;
			float * info = sums+28;
			for(uint i = 0; i < 32; ++i) sums[i] = 0;

			for(uint y = blockIndex; y < size.y; y += 8) {
				for(uint x = sline; x < size.x; x += 112) {
					const TrackData & row = J[(x + y * Jsize.x)];

					if(row.result < 1) {
						info[1] += row.result == -4 ? 1 : 0;
						info[2] += row.result == -5 ? 1 : 0;
						info[3] += row.result > -4 ? 1 : 0;
						continue;
					}
					// Error part
					sums[0] += row.error * row.error;

					// JTe part
					for(int i = 0; i < 6; ++i)
						sums[i+1] += row.error * row.J[i];

					// JTJ part, unfortunatly the double loop is not unrolled well...
					jtj[0] += row.J[0] * row.J[0];
					jtj[1] += row.J[0] * row.J[1];
					jtj[2] += row.J[0] * row.J[2];
					jtj[3] += row.J[0] * row.J[3];

					jtj[4] += row.J[0] * row.J[4];
					jtj[5] += row.J[0] * row.J[5];

					jtj[6] += row.J[1] * row.J[1];
					jtj[7] += row.J[1] * row.J[2];
					jtj[8] += row.J[1] * row.J[3];
					jtj[9] += row.J[1] * row.J[4];

					jtj[10] += row.J[1] * row.J[5];

					jtj[11] += row.J[2] * row.J[2];
					jtj[12] += row.J[2] * row.J[3];
					jtj[13] += row.J[2] * row.J[4];
					jtj[14] += row.J[2] * row.J[5];

					jtj[15] += row.J[3] * row.J[3];
					jtj[16] += row.J[3] * row.J[4];
					jtj[17] += row.J[3] * row.J[5];

					jtj[18] += row.J[4] * row.J[4];
					jtj[19] += row.J[4] * row.J[5];

					jtj[20] += row.J[5] * row.J[5];

					// extra info here
					info[0] += 1;
				}
			}

			for(int i = 0; i < 32; ++i) { // copy over to shared memory
				S[sline][i] = sums[i];
			}
			// WE NO LONGER NEED TO DO THIS AS the threads execute sequentially inside a for loop

		}

		for(int ssline = 0; ssline < 32; ssline++) { // sum up columns and copy to global memory in the final 32 threads
			for(unsigned i = 1; i < 112; ++i) {
				S[0][ssline] += S[i][ssline];
			}
			out[ssline+blockIndex*32] = S[0][ssline];
		}
	}

	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
	for (int j = 1; j < 8; ++j) {
		values[0] += values[j];
	}
	TOCK("reduceKernel", 512);
}
#endif


#ifdef TR_SW_APPROX
void trackKernel(TrackData* output, const float3* inVertex,
		const float3* inNormal, uint2 inSize, const float3* refVertex,
		const float3* refNormal, uint2 refSize, const sMatrix4 Ttrack,
		const sMatrix4 view, const float dist_threshold,
		const float normal_threshold) {
	TICK();
	uint2 pixel = make_uint2(0, 0);
	unsigned int pixely, pixelx;

	#pragma omp parallel for \
			shared(output), private(pixel,pixelx,pixely)
	for (pixely = 0; pixely < inSize.y; pixely+=4) {
		for (pixelx = 0; pixelx < inSize.x; pixelx+=8) {
			pixel.x = pixelx;
			pixel.y = pixely;

			TrackData & c0_row = output[pixel.x + pixel.y * refSize.x];
			TrackData & c0_row_2 = output[pixel.x + 1 + pixel.y * refSize.x];
			TrackData & c0_row_3 = output[pixel.x + 2 + pixel.y * refSize.x];
			TrackData & c0_row_4 = output[pixel.x + 3 + pixel.y * refSize.x];
			TrackData & c0_row_5 = output[pixel.x + 4 + pixel.y * refSize.x];
			TrackData & c0_row_6 = output[pixel.x + 5 + pixel.y * refSize.x];
			TrackData & c0_row_7 = output[pixel.x + 6 + pixel.y * refSize.x];
			TrackData & c0_row_8 = output[pixel.x + 7 + pixel.y * refSize.x];

			TrackData & c1_row = output[pixel.x + (pixel.y + 1) * refSize.x];
			TrackData & c1_row_2 = output[pixel.x + 1 + (pixel.y + 1) * refSize.x];
			TrackData & c1_row_3 = output[pixel.x + 2 + (pixel.y + 1) * refSize.x];
			TrackData & c1_row_4 = output[pixel.x + 3 + (pixel.y + 1) * refSize.x];
			TrackData & c1_row_5 = output[pixel.x + 4 + (pixel.y + 1) * refSize.x];
			TrackData & c1_row_6 = output[pixel.x + 5 + (pixel.y + 1) * refSize.x];
			TrackData & c1_row_7 = output[pixel.x + 6 + (pixel.y + 1) * refSize.x];
			TrackData & c1_row_8 = output[pixel.x + 7 + (pixel.y + 1) * refSize.x];

			TrackData & c2_row = output[pixel.x + (pixel.y + 2) * refSize.x];
			TrackData & c2_row_2 = output[pixel.x + 1 + (pixel.y + 2) * refSize.x];
			TrackData & c2_row_3 = output[pixel.x + 2 + (pixel.y + 2) * refSize.x];
			TrackData & c2_row_4 = output[pixel.x + 3 + (pixel.y + 2) * refSize.x];
			TrackData & c2_row_5 = output[pixel.x + 4 + (pixel.y + 2) * refSize.x];
			TrackData & c2_row_6 = output[pixel.x + 5 + (pixel.y + 2) * refSize.x];
			TrackData & c2_row_7 = output[pixel.x + 6 + (pixel.y + 2) * refSize.x];
			TrackData & c2_row_8 = output[pixel.x + 7 + (pixel.y + 2) * refSize.x];
			
			TrackData & c3_row = output[pixel.x + (pixel.y + 3) * refSize.x];
			TrackData & c3_row_2 = output[pixel.x + 1 + (pixel.y + 3) * refSize.x];
			TrackData & c3_row_3 = output[pixel.x + 2 + (pixel.y + 3) * refSize.x];
			TrackData & c3_row_4 = output[pixel.x + 3 + (pixel.y + 3) * refSize.x];
			TrackData & c3_row_5 = output[pixel.x + 4 + (pixel.y + 3) * refSize.x];
			TrackData & c3_row_6 = output[pixel.x + 5 + (pixel.y + 3) * refSize.x];
			TrackData & c3_row_7 = output[pixel.x + 6 + (pixel.y + 3) * refSize.x];
			TrackData & c3_row_8 = output[pixel.x + 7 + (pixel.y + 3) * refSize.x];

			if (inNormal[pixel.x + pixel.y * inSize.x].x == KFUSION_INVALID) {
				c0_row.result = -1;
				c0_row_2.result = -1;
				c0_row_3.result = -1;
				c0_row_4.result = -1;
				c0_row_5.result = -1;
				c0_row_6.result = -1;
				c0_row_7.result = -1;
				c0_row_8.result = -1;

				c1_row.result = -1;
				c1_row_2.result = -1;
				c1_row_3.result = -1;
				c1_row_4.result = -1;
				c1_row_5.result = -1;
				c1_row_6.result = -1;
				c1_row_7.result = -1;
				c1_row_8.result = -1;

				c2_row.result = -1;
				c2_row_2.result = -1;
				c2_row_3.result = -1;
				c2_row_4.result = -1;
				c2_row_5.result = -1;
				c2_row_6.result = -1;
				c2_row_7.result = -1;
				c2_row_8.result = -1;

				c3_row.result = -1;
				c3_row_2.result = -1;
				c3_row_3.result = -1;
				c3_row_4.result = -1;
				c3_row_5.result = -1;
				c3_row_6.result = -1;
				c3_row_7.result = -1;
				c3_row_8.result = -1;
				continue;
			}

			const float3 projectedVertex = Ttrack
					* inVertex[pixel.x + pixel.y * inSize.x];
			const float3 projectedPos = view * projectedVertex;
			const float2 projPixel = make_float2(
					projectedPos.x / projectedPos.z + 0.5f,
					projectedPos.y / projectedPos.z + 0.5f);
			if (projPixel.x < 0 || projPixel.x > refSize.x - 1
					|| projPixel.y < 0 || projPixel.y > refSize.y - 1) {
				c0_row.result = -2;
				c0_row_2.result = -2;
				c0_row_3.result = -2;
				c0_row_3.result = -2;
				c0_row_4.result = -2;
				c0_row_5.result = -2;
				c0_row_6.result = -2;
				c0_row_7.result = -2;

				c1_row.result = -2;
				c1_row_2.result = -2;
				c1_row_3.result = -2;
				c1_row_3.result = -2;
				c1_row_4.result = -2;
				c1_row_5.result = -2;
				c1_row_6.result = -2;
				c1_row_7.result = -2;

				c2_row.result = -2;
				c2_row_2.result = -2;
				c2_row_3.result = -2;
				c2_row_3.result = -2;
				c2_row_4.result = -2;
				c2_row_5.result = -2;
				c2_row_6.result = -2;
				c2_row_7.result = -2;

				c3_row.result = -2;
				c3_row_2.result = -2;
				c3_row_3.result = -2;
				c3_row_3.result = -2;
				c3_row_4.result = -2;
				c3_row_5.result = -2;
				c3_row_6.result = -2;
				c3_row_7.result = -2;
				continue;
			}

			const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
			const float3 referenceNormal = refNormal[refPixel.x
					+ refPixel.y * refSize.x];

			if (referenceNormal.x == KFUSION_INVALID) {
				c0_row.result = -3;
				c0_row_2.result = -3;
				c0_row_3.result = -3;
				c0_row_4.result = -3;
				c0_row_5.result = -3;
				c0_row_6.result = -3;
				c0_row_7.result = -3;
				c0_row_8.result = -3;

				c1_row.result = -3;
				c1_row_2.result = -3;
				c1_row_3.result = -3;
				c1_row_4.result = -3;
				c1_row_5.result = -3;
				c1_row_6.result = -3;
				c1_row_7.result = -3;
				c1_row_8.result = -3;

				c2_row.result = -3;
				c2_row_2.result = -3;
				c2_row_3.result = -3;
				c2_row_4.result = -3;
				c2_row_5.result = -3;
				c2_row_6.result = -3;
				c2_row_7.result = -3;
				c2_row_8.result = -3;

				c3_row.result = -3;
				c3_row_2.result = -3;
				c3_row_3.result = -3;
				c3_row_4.result = -3;
				c3_row_5.result = -3;
				c3_row_6.result = -3;
				c3_row_7.result = -3;
				c3_row_8.result = -3;
				continue;
			}

			const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x]
					- projectedVertex;
			const float3 projectedNormal = rotate(Ttrack,
					inNormal[pixel.x + pixel.y * inSize.x]);

			if (length(diff) > dist_threshold) {
				c0_row.result = -4;
				c0_row_2.result = -4;
				c0_row_3.result = -4;
				c0_row_4.result = -4;
				c0_row_5.result = -4;
				c0_row_6.result = -4;
				c0_row_7.result = -4;
				c0_row_8.result = -4;

				c1_row.result = -4;
				c1_row_2.result = -4;
				c1_row_3.result = -4;
				c1_row_4.result = -4;
				c1_row_5.result = -4;
				c1_row_6.result = -4;
				c1_row_7.result = -4;
				c1_row_8.result = -4;

				c2_row.result = -4;
				c2_row_2.result = -4;
				c2_row_3.result = -4;
				c2_row_4.result = -4;
				c2_row_5.result = -4;
				c2_row_6.result = -4;
				c2_row_7.result = -4;
				c2_row_8.result = -4;

				c3_row.result = -4;
				c3_row_2.result = -4;
				c3_row_3.result = -4;
				c3_row_4.result = -4;
				c3_row_5.result = -4;
				c3_row_6.result = -4;
				c3_row_7.result = -4;
				c3_row_8.result = -4;
				continue;
			}
			if (dot(projectedNormal, referenceNormal) < normal_threshold) {
				c0_row.result = -5;
				c0_row_2.result = -5;
				c0_row_3.result = -5;
				c0_row_4.result = -5;
				c0_row_5.result = -5;
				c0_row_6.result = -5;
				c0_row_7.result = -5;
				c0_row_8.result = -5;

				c1_row.result = -5;
				c1_row_2.result = -5;
				c1_row_3.result = -5;
				c1_row_4.result = -5;
				c1_row_5.result = -5;
				c1_row_6.result = -5;
				c1_row_7.result = -5;
				c1_row_8.result = -5;

				c2_row.result = -5;
				c2_row_2.result = -5;
				c2_row_3.result = -5;
				c2_row_4.result = -5;
				c2_row_5.result = -5;
				c2_row_6.result = -5;
				c2_row_7.result = -5;
				c2_row_8.result = -5;

				c3_row.result = -5;
				c3_row_2.result = -5;
				c3_row_3.result = -5;
				c3_row_4.result = -5;
				c3_row_5.result = -5;
				c3_row_6.result = -5;
				c3_row_7.result = -5;
				c3_row_8.result = -5;
				continue;
			}
			c0_row.result = 1;
			c0_row_2.result = 1;
			c0_row_3.result = 1;
			c0_row_4.result = 1;
			c0_row_5.result = 1;
			c0_row_6.result = 1;
			c0_row_7.result = 1;
			c0_row_8.result = 1;

			c1_row.result = 1;
			c1_row_2.result = 1;
			c1_row_3.result = 1;
			c1_row_4.result = 1;
			c1_row_5.result = 1;
			c1_row_6.result = 1;
			c1_row_7.result = 1;
			c1_row_8.result = 1;

			c2_row.result = 1;
			c2_row_2.result = 1;
			c2_row_3.result = 1;
			c2_row_4.result = 1;
			c2_row_5.result = 1;
			c2_row_6.result = 1;
			c2_row_7.result = 1;
			c2_row_8.result = 1;

			c3_row.result = 1;
			c3_row_2.result = 1;
			c3_row_3.result = 1;
			c3_row_4.result = 1;
			c3_row_5.result = 1;
			c3_row_6.result = 1;
			c3_row_7.result = 1;
			c3_row_8.result = 1;

			c0_row.error = dot(referenceNormal, diff);
			c0_row_2.error = c1_row.error;
			c0_row_3.error = c1_row.error;
			c0_row_4.error = c1_row.error;
			c0_row_5.error = c1_row.error;
			c0_row_6.error = c1_row.error;
			c0_row_7.error = c1_row.error;
			c0_row_8.error = c1_row.error;

			c1_row.error = c0_row.error;
			c1_row_2.error = c1_row.error;
			c1_row_3.error = c1_row.error;
			c1_row_4.error = c1_row.error;
			c1_row_5.error = c1_row.error;
			c1_row_6.error = c1_row.error;
			c1_row_7.error = c1_row.error;
			c1_row_8.error = c1_row.error;

			c2_row.error = c0_row.error;
			c2_row_2.error = c1_row.error;
			c2_row_3.error = c1_row.error;
			c2_row_4.error = c1_row.error;
			c2_row_5.error = c1_row.error;
			c2_row_6.error = c1_row.error;
			c2_row_7.error = c1_row.error;
			c2_row_8.error = c1_row.error;

			c3_row.error = c0_row.error;
			c3_row_2.error = c1_row.error;
			c3_row_3.error = c1_row.error;
			c3_row_4.error = c1_row.error;
			c3_row_5.error = c1_row.error;
			c3_row_6.error = c1_row.error;
			c3_row_7.error = c1_row.error;
			c3_row_8.error = c1_row.error;

			float3 scnd = cross(projectedVertex, referenceNormal);
			((float3 *) c0_row.J)[0] = referenceNormal;
			((float3 *) c0_row.J)[1] = scnd;

			((float3 *) c0_row_2.J)[0] = referenceNormal;
			((float3 *) c0_row_2.J)[1] = scnd;

			((float3 *) c0_row_3.J)[0] = referenceNormal;
			((float3 *) c0_row_3.J)[1] = scnd;
		
			((float3 *) c0_row_4.J)[0] = referenceNormal;
			((float3 *) c0_row_4.J)[1] = scnd;
		
			((float3 *) c0_row_5.J)[0] = referenceNormal;
			((float3 *) c0_row_5.J)[1] = scnd;
		
			((float3 *) c0_row_6.J)[0] = referenceNormal;
			((float3 *) c0_row_6.J)[1] = scnd;
		
			((float3 *) c0_row_7.J)[0] = referenceNormal;
			((float3 *) c0_row_7.J)[1] = scnd;
		
			((float3 *) c0_row_8.J)[0] = referenceNormal;
			((float3 *) c0_row_8.J)[1] = scnd;

			((float3 *) c1_row.J)[0] = referenceNormal;
			((float3 *) c1_row.J)[1] = scnd;
			
			((float3 *) c1_row_2.J)[0] = referenceNormal;
			((float3 *) c1_row_2.J)[1] = scnd;

			((float3 *) c1_row_3.J)[0] = referenceNormal;
			((float3 *) c1_row_3.J)[1] = scnd;
		
			((float3 *) c1_row_4.J)[0] = referenceNormal;
			((float3 *) c1_row_4.J)[1] = scnd;
		
			((float3 *) c1_row_5.J)[0] = referenceNormal;
			((float3 *) c1_row_5.J)[1] = scnd;
		
			((float3 *) c1_row_6.J)[0] = referenceNormal;
			((float3 *) c1_row_6.J)[1] = scnd;
		
			((float3 *) c1_row_7.J)[0] = referenceNormal;
			((float3 *) c1_row_7.J)[1] = scnd;
		
			((float3 *) c1_row_8.J)[0] = referenceNormal;
			((float3 *) c1_row_8.J)[1] = scnd;

			((float3 *) c2_row.J)[0] = referenceNormal;
			((float3 *) c2_row.J)[1] = scnd;
			
			((float3 *) c2_row_2.J)[0] = referenceNormal;
			((float3 *) c2_row_2.J)[1] = scnd;

			((float3 *) c2_row_3.J)[0] = referenceNormal;
			((float3 *) c2_row_3.J)[1] = scnd;
		
			((float3 *) c2_row_4.J)[0] = referenceNormal;
			((float3 *) c2_row_4.J)[1] = scnd;
		
			((float3 *) c2_row_5.J)[0] = referenceNormal;
			((float3 *) c2_row_5.J)[1] = scnd;
		
			((float3 *) c2_row_6.J)[0] = referenceNormal;
			((float3 *) c2_row_6.J)[1] = scnd;
		
			((float3 *) c2_row_7.J)[0] = referenceNormal;
			((float3 *) c2_row_7.J)[1] = scnd;
		
			((float3 *) c2_row_8.J)[0] = referenceNormal;
			((float3 *) c2_row_8.J)[1] = scnd;

			((float3 *) c3_row.J)[0] = referenceNormal;
			((float3 *) c3_row.J)[1] = scnd;
			
			((float3 *) c3_row_2.J)[0] = referenceNormal;
			((float3 *) c3_row_2.J)[1] = scnd;

			((float3 *) c3_row_3.J)[0] = referenceNormal;
			((float3 *) c3_row_3.J)[1] = scnd;
		
			((float3 *) c3_row_4.J)[0] = referenceNormal;
			((float3 *) c3_row_4.J)[1] = scnd;
		
			((float3 *) c3_row_5.J)[0] = referenceNormal;
			((float3 *) c3_row_5.J)[1] = scnd;
		
			((float3 *) c3_row_6.J)[0] = referenceNormal;
			((float3 *) c3_row_6.J)[1] = scnd;
		
			((float3 *) c3_row_7.J)[0] = referenceNormal;
			((float3 *) c3_row_7.J)[1] = scnd;
		
			((float3 *) c3_row_8.J)[0] = referenceNormal;
			((float3 *) c3_row_8.J)[1] = scnd;
		
		}
	}
	TOCK("trackKernel", inSize.x * inSize.y);
}
#else
void trackKernel(TrackData* output, const float3* inVertex,
		const float3* inNormal, uint2 inSize, const float3* refVertex,
		const float3* refNormal, uint2 refSize, const sMatrix4 Ttrack,
		const sMatrix4 view, const float dist_threshold,
		const float normal_threshold) {
	TICK();
	uint2 pixel = make_uint2(0, 0);
	unsigned int pixely, pixelx;

	#pragma omp parallel for \
			shared(output), private(pixel,pixelx,pixely)
	for (pixely = 0; pixely < inSize.y; pixely++) {
		for (pixelx = 0; pixelx < inSize.x; pixelx++) {
			pixel.x = pixelx;
			pixel.y = pixely;

			TrackData & row = output[pixel.x + pixel.y * refSize.x];

			if (inNormal[pixel.x + pixel.y * inSize.x].x == KFUSION_INVALID) {
				row.result = -1;
				continue;
			}

			const float3 projectedVertex = Ttrack
					* inVertex[pixel.x + pixel.y * inSize.x];
			const float3 projectedPos = view * projectedVertex;
			const float2 projPixel = make_float2(
					projectedPos.x / projectedPos.z + 0.5f,
					projectedPos.y / projectedPos.z + 0.5f);
			if (projPixel.x < 0 || projPixel.x > refSize.x - 1
					|| projPixel.y < 0 || projPixel.y > refSize.y - 1) {
				row.result = -2;
				continue;
			}

			const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
			const float3 referenceNormal = refNormal[refPixel.x
					+ refPixel.y * refSize.x];

			if (referenceNormal.x == KFUSION_INVALID) {
				row.result = -3;
				continue;
			}

			const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x]
					- projectedVertex;
			const float3 projectedNormal = rotate(Ttrack,
					inNormal[pixel.x + pixel.y * inSize.x]);

			if (length(diff) > dist_threshold) {
				row.result = -4;
				continue;
			}
			if (dot(projectedNormal, referenceNormal) < normal_threshold) {
				row.result = -5;
				continue;
			}
			row.result = 1;
			row.error = dot(referenceNormal, diff);
			((float3 *) row.J)[0] = referenceNormal;
			((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
		}
	}
	TOCK("trackKernel", inSize.x*inSize.y);
}
#endif

#ifdef PREF_FUSEVOL
#ifndef FUSE_HW
void fuseVolumesKernelPrefetched(	Volume dstVol, 
									float *tsdfBuff, 
									const sMatrix4 pose,
									const float3 origin,
									const float maxweight)
{
	TICK();
	
	unsigned int z;

	#pragma omp parallel for \
			shared(dstVol), private(z)
	for (z = 0; z < dstVol._resolution.z; z+=FUSE_LP) {
		int z_idx = z * dstVol._resolution.x*dstVol._resolution.y;

		for (unsigned int y = 0; y < dstVol._resolution.y; y++) {
			int y_idx = y*dstVol._resolution.x;

			for (unsigned int x = 0; x < dstVol._resolution.x; x++) {
				uint3 pix = make_uint3(x, y, z);
				
				float tsdf = tsdfBuff[x + y_idx + z_idx];
				if(tsdf == 1.0)
					continue;
				float w_interp = 1;
				
				float2 p_data = dstVol[pix];

				float w = p_data.y;
				float new_w = w + w_interp;
				
				p_data.x = clamp( (w*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
				p_data.y = fminf(new_w, maxweight);
				
				dstVol.set(pix, p_data);
				
			}
		}
	}

	TOCK("fuseVolumesKernel", dstVol.getResolution().y*dstVol.getResolution().x);
}
#endif
#endif



void fuseVolumesKernel( Volume dstVol, 
						Volume srcVol, 
						const sMatrix4 pose,
						const float3 origin,
						const float maxweight)
{
	TICK();
	
	
	float3 vsize=srcVol.getSizeInMeters();    
	
	unsigned int z;

	#pragma omp parallel for \
			shared(dstVol), private(z)
	for (z = 0; z < dstVol.getResolution().z; z++) {  
		for (unsigned int y = 0; y < dstVol.getResolution().y; y++) {
			for (unsigned int x = 0; x < dstVol.getResolution().x; x++) {
				uint3 pix = make_uint3(x, y, z);
				float3 pos = dstVol.pos(pix);
				
				pos = pose*pos;
				
				pos.x += origin.x;
				pos.y += origin.y;
				pos.z += origin.z;
				
				if( pos.x < 0 || pos.x >= vsize.x ||
					pos.y < 0 || pos.y >= vsize.y ||
					pos.z < 0 || pos.z >= vsize.z)
				{
					 continue;
				}
				
				float tsdf = srcVol.interp(pos);
				float w_interp = 1;
				
				float2 p_data = dstVol[pix];
				
				if(tsdf == 1.0)
					continue;
				
				float w = p_data.y;
				float new_w = w + w_interp;
				
				p_data.x = clamp( (w*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
				p_data.y = fminf(new_w, maxweight);                
				
				dstVol.set(pix, p_data);
				
			}
		}
	}

	TOCK("fuseVolumesKernel", dstVol.getResolution().y*dstVol.getResolution().x);
}

#ifdef INTKF // Enabled when integrateKeyFrameData calls Integrate HW accelerators according with the INT_* optimizations.

int startFlag_kf = -1;

void KFusion::integrateKernel(Volume vol, const float* depth, const uchar3* rgb, uint2 depthSize,
		const sMatrix4 invTrack, const sMatrix4 K, const float mu,
		const float maxweight, const int ratio) {
	TICK();
	
	int argcounter = 0;
	int offset;
	int end;

	int initFlag = 0;
	if (prefetch_kfvol_flag != 0) {
		initFlag = 1;
		prefetch_kfvol_flag = 0;
	}

	#ifdef INT_LP4
		startFlag_kf++;
	#else
		#ifdef INT_LP6
		startFlag_kf++;
		#endif
	#endif

	#ifdef INT_NCU
		cl_event buffDone[CU], krnlDone[CU], flagDone[CU];
		cl_event event;

		#pragma omp parallel for
		for (int i = 0; i < CU; i++) {
			#ifdef INTKF_UNIQUEUE
				pt_intkf_in[i][0] = volSize_kf_buffer[i];
				pt_intkf_in[i][1] = volConst_kf_buffer[i];
				pt_intkf_in[i][2] = volDim_kf_buffer[i];
				pt_intkf_in[i][3] = integrate_depth_kf_buffer[i];
				pt_intkf_in[i][4] = InvTrack_data_kf_buffer[i];
				pt_intkf_in[i][5] = K_data_kf_buffer[i];
			#else
				pt_intkf_in[i][0] = volSize_buffer[i];
				pt_intkf_in[i][1] = volConst_buffer[i];
				pt_intkf_in[i][2] = volDim_buffer[i];
				pt_intkf_in[i][3] = integrate_depth_buffer[i];
				pt_intkf_in[i][4] = InvTrack_data_buffer[i];
				pt_intkf_in[i][5] = K_data_buffer[i];
			#endif
		}

		#ifdef INTKF_UNIQUEUE
			int j;
			// before migration
			#pragma omp parallel for
			for (j = 0; j < CU; j++)
				for (int i = 0; i < 4; i ++) {
					InvTrack_data_kf[j][i*4] = invTrack.data[i].x;
					InvTrack_data_kf[j][i*4 + 1] = invTrack.data[i].y;
					InvTrack_data_kf[j][i*4 + 2] = invTrack.data[i].z;
					InvTrack_data_kf[j][i*4 + 3] = invTrack.data[i].w;

					K_data_kf[j][i*4] = K.data[i].x;
					K_data_kf[j][i*4 + 1] = K.data[i].y;
					K_data_kf[j][i*4 + 2] = K.data[i].z;
					K_data_kf[j][i*4 + 3] = K.data[i].w;
				}
		#endif

		#ifdef INT_HP
			floatH mu_h = mu;
			floatH maxweight_h = maxweight;

			#ifdef INTKF_UNIQUEUE
				int k;
				#pragma omp parallel for
				for (k = 0; k < CU; k++)
					memcpy(floatDepth_kf[k], floatDepth[k], depthSize.x*depthSize.y*sizeof(floatH));
			#endif
		#else
			#ifdef INTKF_UNIQUEUE
				for (int i = 0; i < CU; i++) 
					memcpy(floatDepth_kf[i], depth, depthSize.x*depthSize.y*sizeof(float));
			#endif
		#endif
	
		#ifdef INT_LP6
			int startFlag_offset = 0;
			int startFlag_krnl = startFlag_kf;
		#endif

		end = vol._resolution.z/CU;

		for (int i = 0; i < CU; i++) {
			argcounter = 0;
			offset = i*end;;

			err = 0;
			#ifdef INTKF_UNIQUEUE
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volSize_kf_buffer[i]);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volConst_kf_buffer[i]);
			#else
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volSize_buffer[i]);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volConst_buffer[i]);
			#endif
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_kf_vol_sbuffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_kf_vol_sbuffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_kf_vol_sbuffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_kf_vol_sbuffer[i]);
			#ifdef INTKF_UNIQUEUE
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volDim_kf_buffer[i]);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_depth_kf_buffer[i]);
			#else
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volDim_buffer[i]);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_depth_buffer[i]);
			#endif
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &depthSize.x);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &depthSize.y);
			#ifdef INTKF_UNIQUEUE
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &InvTrack_data_kf_buffer[i]);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &K_data_kf_buffer[i]);
			#else
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &InvTrack_data_buffer[i]);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &K_data_buffer[i]);
			#endif
			#ifdef INT_HP
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(floatH), &mu_h);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(floatH), &maxweight_h);
			#else
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(float), &mu);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(float), &maxweight);
			#endif
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(uint), &offset);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &end);
			#ifdef INT_LP6
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(uint), &startFlag_krnl);
			#else
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(uint), &startFlag_kf);
			#endif
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &initFlag);
			
			#ifdef INTKF_UNIQUEUE
				err |= clEnqueueMigrateMemObjects(q_intkf,(cl_uint)5, pt_intkf_in[i], 0 ,0,NULL, NULL);
			#else
				err |= clEnqueueMigrateMemObjects(q,(cl_uint)5, pt_intkf_in[i], 0 ,0,NULL, NULL);
			#endif
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set krnl_integrateKernel arguments! %d\n", err);
			}

			const cl_mem buf_in = integrate_kf_vol_sbuffer[i];
			#ifdef INTKF_UNIQUEUE
				err = clEnqueueMigrateMemObjects(q_intkf,(cl_uint)1, &buf_in, 0, 0, NULL, &event);
			#else
				err = clEnqueueMigrateMemObjects(q,(cl_uint)1, &buf_in, 0, 0, NULL, &event);
			#endif
			if (err != CL_SUCCESS) {
				printf("Error: Failed to input integrate_kf_vol_sbuffer[%d]! %d\n", i, err);
				exit(EXIT_FAILURE);
			}
			buffDone[i] = event;

			#ifdef INT_LP6
				if (CU == 4)
					startFlag_offset += 2;
				if (CU == 2)
					startFlag_offset += 4;

				startFlag_krnl = startFlag_kf + startFlag_offset;
			#endif
		}

		for (int i = 0; i < CU; i++) {
			#ifdef INTKF_UNIQUEUE
				err = clEnqueueTask(q_intkf, krnl_integrateKernel[i], 1, &buffDone[i], &krnlDone[i]);
			#else
				err = clEnqueueTask(q, krnl_integrateKernel[i], 1, &buffDone[i], &krnlDone[i]);
			#endif
			if (err) {
				printf("Error: Failed to execute integrate kernel[%d]! %d\n", i, err);
				exit(EXIT_FAILURE);
			}
		}

		#ifdef R_INTERLEAVING
			#ifdef R_RATE
			if (_frame % RAYCAST_RATE == 3)
			#endif
				raycasting(_frame);
		#endif

		for (int i = 0; i < CU; i++) {
			err = 0;
			const cl_mem buf_in = integrate_vol_sbuffer[i];

			#ifdef INTKF_UNIQUEUE
				err = clEnqueueMigrateMemObjects(q_intkf,(cl_uint)1, &buf_in, CL_MIGRATE_MEM_OBJECT_HOST, 1, &krnlDone[i], &flagDone[i]);
			#else
				err = clEnqueueMigrateMemObjects(q,(cl_uint)1, &buf_in, CL_MIGRATE_MEM_OBJECT_HOST, 1, &krnlDone[i], &flagDone[i]);
			#endif
			if (err != CL_SUCCESS) {
				printf("Error: Failed to execute kernel! %d\n", err);
				exit(EXIT_FAILURE);
			}
		}

		clWaitForEvents(CU, flagDone);

	#else  // not INT_NCU
		#ifdef INTKF_UNIQUEUE
			pt_intkf_in[0] = volSize_kf_buffer;
			pt_intkf_in[1] = volConst_kf_buffer;
			pt_intkf_in[2] = volDim_kf_buffer;
			pt_intkf_in[3] = integrate_depth_kf_buffer;
			pt_intkf_in[4] = InvTrack_data_kf_buffer;
			pt_intkf_in[5] = K_data_kf_buffer;
		#else
			pt_intkf_in[0] = volSize_buffer;
			pt_intkf_in[1] = volConst_buffer;
			pt_intkf_in[2] = volDim_buffer;
			pt_intkf_in[3] = integrate_depth_buffer;
			pt_intkf_in[4] = InvTrack_data_buffer;
			pt_intkf_in[5] = K_data_buffer;
		#endif

		#ifdef INTKF_UNIQUEUE
			int i;
			#pragma omp parallel for
			for (i = 0; i < 4; i ++) {
				InvTrack_data_kf[i*4] = invTrack.data[i].x;
				InvTrack_data_kf[i*4 + 1] = invTrack.data[i].y;
				InvTrack_data_kf[i*4 + 2] = invTrack.data[i].z;
				InvTrack_data_kf[i*4 + 3] = invTrack.data[i].w;

				K_data_kf[i*4] = K.data[i].x;
				K_data_kf[i*4 + 1] = K.data[i].y;
				K_data_kf[i*4 + 2] = K.data[i].z;
				K_data_kf[i*4 + 3] = K.data[i].w;
			}
		#endif

		#ifdef INT_HP
			floatH mu_h = mu;
			floatH maxweight_h = maxweight;

			#ifdef INTKF_UNIQUEUE
			memcpy(floatDepth_kf, floatDepth, depthSize.x*depthSize.y*sizeof(floatH));
			#endif
		#endif

		end = vol._resolution.z;
		
		argcounter = 0;
		offset = 0;

		err = 0;
		#ifdef INTKF_UNIQUEUE
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volSize_kf_buffer);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volConst_kf_buffer);
		#else
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volSize_buffer);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volConst_buffer);
		#endif
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_kf_vol_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_kf_vol_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_kf_vol_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_kf_vol_buffer);
		#ifdef INTKF_UNIQUEUE
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volDim_kf_buffer);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_depth_kf_buffer);
		#else
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volDim_buffer);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_depth_buffer);
		#endif
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &depthSize.x);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &depthSize.y);
		#ifdef INTKF_UNIQUEUE
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &InvTrack_data_kf_buffer);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &K_data_kf_buffer);
		#else
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &InvTrack_data_buffer);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &K_data_buffer);
		#endif
		#ifdef INT_HP
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(floatH), &mu_h);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(floatH), &maxweight_h);
		#else
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(float), &mu);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(float), &maxweight);
		#endif
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(uint), &offset);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &end);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(uint), &startFlag_kf);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &initFlag);
		
		#ifdef INTKF_UNIQUEUE
			err |= clEnqueueMigrateMemObjects(q_intkf,(cl_uint)5, pt_intkf_in, 0 ,0,NULL, NULL);
		#else
			err |= clEnqueueMigrateMemObjects(q,(cl_uint)5, pt_intkf_in, 0 ,0,NULL, NULL);
		#endif
		if (err != CL_SUCCESS) {
			printf("Error: Failed to set krnl_integrateKernel arguments! %d\n", err);

		}

		err = 0;
		const cl_mem buf_in = integrate_kf_vol_buffer;

		#ifdef INTKF_UNIQUEUE
			err = clEnqueueMigrateMemObjects(q_intkf, (cl_uint)1, &buf_in, 0 ,0,NULL, NULL);
		#else
			err = clEnqueueMigrateMemObjects(q, (cl_uint)1, &buf_in, 0 ,0,NULL, NULL);
		#endif
		if (err) {
			printf("Error: Failed to input clEnqueueMigrateMemObjects! %d\n", err);
			exit(EXIT_FAILURE);
		}

		#ifdef INTKF_UNIQUEUE
			err = clEnqueueTask(q_intkf, krnl_integrateKernel, 0, NULL, NULL);
		#else
			err = clEnqueueTask(q, krnl_integrateKernel, 0, NULL, NULL);
		#endif
		if (err) {
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(EXIT_FAILURE);
		}

		#ifdef R_INTERLEAVING
			#ifdef R_RATE
			if (_frame % RAYCAST_RATE == 3)
			#endif
				raycasting(_frame);
		#endif

		err = 0;
		
		#ifdef INTKF_UNIQUEUE
			clFinish(q_intkf);
			err = clEnqueueMigrateMemObjects(q_intkf,(cl_uint)1, &buf_in, CL_MIGRATE_MEM_OBJECT_HOST,0,NULL, NULL);
		#else
			clFinish(q);
			err = clEnqueueMigrateMemObjects(q,(cl_uint)1, &buf_in, CL_MIGRATE_MEM_OBJECT_HOST,0,NULL, NULL);
		#endif

		if (err) {
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(EXIT_FAILURE);
		}

		#ifdef INTKF_UNIQUEUE
			clFinish(q_intkf);
		#else
			clFinish(q);
		#endif
	#endif // INT_NCU

	TOCK("integrateKernel", vol._resolution.x*vol._resolution.y*vol._resolution.z);
}

#else

void KFusion::integrateKernel(Volume vol, const float* depth, const uchar3* rgb, uint2 depthSize,
		const sMatrix4 invTrack, const sMatrix4 K, const float mu,
		const float maxweight, const int ratio) {
	TICK();
	const float3 delta = rotate(invTrack,
			make_float3(0, 0, vol.dim.z / vol._resolution.z));
	const float3 cameraDelta = rotate(K, delta);
	unsigned int y;

	#pragma omp parallel for \
		shared(vol), private(y)
	for (y = 0; y < vol._resolution.y; y++)
		for (unsigned int x = 0; x < vol._resolution.x; x++) {

			uint3 pix = make_uint3(x, y, 0);
			float3 pos = invTrack * vol.pos(pix);
			float3 cameraX = K * pos;

			for (pix.z = 0; pix.z < vol._resolution.z;
					++pix.z, pos += delta, cameraX += cameraDelta) {

				if (pos.z < 0.0001f) // some near plane constraint
					continue;

				const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
						cameraX.y / cameraX.z + 0.5f);

				if (pixel.x < 0 || pixel.x > depthSize.x - 1 || pixel.y < 0
						|| pixel.y > depthSize.y - 1)
					continue;

				uint2 px = make_uint2(pixel.x, pixel.y);
				if (depth[px.x + px.y * depthSize.x] == 0 /*|| std::isnan(depth[px.x + px.y * depthSize.x])*/)
					continue;

				const float diff =
						(depth[px.x + px.y * depthSize.x] - cameraX.z)
								* std::sqrt(
										1 + (pos.x / pos.z)*(pos.x / pos.z)
												+ (pos.y / pos.z)*(pos.y / pos.z));
				if (diff > -mu) {
					const float sdf = fminf(1.f, diff / mu);
					float2 data = vol[pix];

					float w = data.y;
					float new_w = w + 1;
					data.x = clamp((w*data.x + sdf)/new_w, -1.f, 1.f);

					data.y = fminf(new_w, maxweight);
					vol.set(pix, data);
				}
			}
		}

	TOCK("integrateKernel", vol._resolution.x*vol._resolution.y*vol._resolution.z);
}
#endif


int startFlag = -1;

void KFusion::integrateHWKernel(Volume vol, const float* depth, const uchar3* rgb, uint2 depthSize,
		const sMatrix4 invTrack, const sMatrix4 K, const float mu,
		const float maxweight, const int ratio) {
	TICK();
	
	int argcounter = 0;
	int offset;
	int end;
	int initFlag = 0;

	err = clRetainCommandQueue(q);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to retain command queue q! %d\n", err);
	}
	#ifdef INT_LP4
		startFlag++;
	#else
		#ifdef INT_LP6
		startFlag++;
		#endif
	#endif

	#ifdef INT_NCU
		cl_event buffDone[CU], krnlDone[CU], flagDone[CU];
		cl_event event;

		#pragma omp parallel for
		for (int i = 0; i < CU; i++) {
			pt_int_in[i][0] = volSize_buffer[i];
			pt_int_in[i][1] = volConst_buffer[i];
			pt_int_in[i][2] = volDim_buffer[i];
			pt_int_in[i][3] = integrate_depth_buffer[i];
			pt_int_in[i][4] = InvTrack_data_buffer[i];
			pt_int_in[i][5] = K_data_buffer[i];
		}

		int j;
		// before migration
		#pragma omp parallel for
		for (j = 0; j < CU; j++)
			for (int i = 0; i < 4; i ++) {
				InvTrack_data[j][i*4] = invTrack.data[i].x;
				InvTrack_data[j][i*4 + 1] = invTrack.data[i].y;
				InvTrack_data[j][i*4 + 2] = invTrack.data[i].z;
				InvTrack_data[j][i*4 + 3] = invTrack.data[i].w;

				K_data[j][i*4] = K.data[i].x;
				K_data[j][i*4 + 1] = K.data[i].y;
				K_data[j][i*4 + 2] = K.data[i].z;
				K_data[j][i*4 + 3] = K.data[i].w;
			}

		#ifdef INT_HP
			floatH mu_h = mu;
			floatH maxweight_h = maxweight;

			int k;
			#pragma omp parallel for
			for (k = 0; k < CU; k++)
				for (int i = 0; i < depthSize.x*depthSize.y; i++)
					floatDepth[k][i] = depth[i];
		#else
			for (int i = 0; i < CU; i++) 
				memcpy(floatDepth[i], depth, depthSize.x*depthSize.y*sizeof(float));
		#endif
	
		#ifdef INT_LP6
			int startFlag_offset = 0;
			int startFlag_krnl = startFlag;
		#endif

		end = vol._resolution.z/CU;

		for (int i = 0; i < CU; i++) {
			argcounter = 0;
			offset = i*end;

			err = 0;
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volSize_buffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volConst_buffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &volDim_buffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &integrate_depth_buffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &depthSize.x);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &depthSize.y);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &InvTrack_data_buffer[i]);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(cl_mem), &K_data_buffer[i]);
			#ifdef INT_HP
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(floatH), &mu_h);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(floatH), &maxweight_h);
			#else
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(float), &mu);
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(float), &maxweight);
			#endif
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(uint), &offset);
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &end);
			#ifdef INT_LP6
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(uint), &startFlag_krnl);
			#else
				err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(uint), &startFlag);
			#endif
			err |= clSetKernelArg(krnl_integrateKernel[i], argcounter++, sizeof(int), &initFlag);
			
			
			err |= clEnqueueMigrateMemObjects(q,(cl_uint)5, pt_int_in[i], 0 ,0,NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set krnl_integrateKernel arguments! %d\n", err);

			}
			const cl_mem buf_in = integrate_vol_sbuffer[i];
			err = clEnqueueMigrateMemObjects(q,(cl_uint)1, &buf_in, 0, 0, NULL, &event);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to input integrate_vol_sbuffer[%d]! %d\n", i, err);
				exit(EXIT_FAILURE);
			}
			buffDone[i] = event;

			#ifdef INT_LP6
				if (CU == 4)
					startFlag_offset += 2;
				if (CU == 2)
					startFlag_offset += 4;

				startFlag_krnl = startFlag + startFlag_offset;
			#endif
		}

		for (int i = 0; i < CU; i++) {		
			err = clEnqueueTask(q, krnl_integrateKernel[i], 1, &buffDone[i], &krnlDone[i]);
			if (err) {
				printf("Error: Failed to execute integrate kernel[%d]! %d\n", i, err);
				exit(EXIT_FAILURE);
			}
		}

		#ifdef PREF_FUSEVOL
		#ifdef POSEGRAPHOPT_ENABLED
			if (posegraphOptEnabled > 5)
		#endif
			if (prefetch_kfvol_flag != 0) {
	
				int idx = ((prefetch_kfvol_flag/params.keyframe_rate) - 1)%params.max_num_kfs;
				// std::cout << "idx of pref frame = " << idx << "\t vol.frame = " << volumes[idx].frame << std::endl;
				#ifndef CHANGE_VOLS
					memcpy(fusionVol.data, volumes[idx].data, fusionVol._resolution.x*fusionVol._resolution.y*fusionVol._resolution.z*sizeof(short2));
				#endif
	
				#ifdef CHANGE_VOLS
					prefetchFuseVolInterp(fuseVol_interp_ptr[idx], keyFrameVol, 
								inverse(volumes[idx].pose), params.volume_direction, 
								fusionVol.voxelSize);
				#else
					prefetchFuseVolInterp(fuseVol_interp_ptr[idx], fusionVol, 
								inverse(volumes[idx].pose), params.volume_direction, 
								fusionVol.voxelSize);
				#endif
				
				#ifndef INTKF
				prefetch_kfvol_flag = 0;
				#endif
				// std::cout << "prefetching done" << std::endl;
			}
		#endif

		for (int i = 0; i < CU; i++) {
			err = 0;
			const cl_mem buf_in = integrate_vol_sbuffer[i];
			err = clEnqueueMigrateMemObjects(q,(cl_uint)1, &buf_in, CL_MIGRATE_MEM_OBJECT_HOST, 1, &krnlDone[i], &flagDone[i]);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to execute kernel! %d\n", err);
				exit(EXIT_FAILURE);
			}
		}

		clWaitForEvents(CU, flagDone);

	#else
		pt_int_in[0] = volSize_buffer;
		pt_int_in[1] = volConst_buffer;
		pt_int_in[2] = volDim_buffer;
		pt_int_in[3] = integrate_depth_buffer;
		pt_int_in[4] = InvTrack_data_buffer;
		pt_int_in[5] = K_data_buffer;

		int i;
		#pragma omp parallel for
		for (i = 0; i < 4; i ++) {
			InvTrack_data[i*4] = invTrack.data[i].x;
			InvTrack_data[i*4 + 1] = invTrack.data[i].y;
			InvTrack_data[i*4 + 2] = invTrack.data[i].z;
			InvTrack_data[i*4 + 3] = invTrack.data[i].w;

			K_data[i*4] = K.data[i].x;
			K_data[i*4 + 1] = K.data[i].y;
			K_data[i*4 + 2] = K.data[i].z;
			K_data[i*4 + 3] = K.data[i].w;
		}

		#ifdef INT_HP
			floatH mu_h = mu;
			floatH maxweight_h = maxweight;

			#pragma omp parallel for
			for (i = 0; i < depthSize.x*depthSize.y; i++)
				floatDepth[i] = depth[i];
		#endif

		end = vol._resolution.z;
		
		argcounter = 0;
		offset = 0;

		err = 0;
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volSize_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volConst_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &volDim_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &integrate_depth_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &depthSize.x);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &depthSize.y);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &InvTrack_data_buffer);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(cl_mem), &K_data_buffer);
		#ifdef INT_HP
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(floatH), &mu_h);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(floatH), &maxweight_h);
		#else
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(float), &mu);
			err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(float), &maxweight);
		#endif
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(uint), &offset);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &end);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(uint), &startFlag);
		err |= clSetKernelArg(krnl_integrateKernel, argcounter++, sizeof(int), &initFlag);
		
		
		err |= clEnqueueMigrateMemObjects(q,(cl_uint)5, pt_int_in, 0 ,0,NULL, NULL);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to set krnl_integrateKernel arguments! %d\n", err);

		}

		err = 0;
		const cl_mem buf_in = integrate_vol_buffer;
		err = clEnqueueMigrateMemObjects(q,(cl_uint)1, &buf_in, 0 ,0,NULL, NULL);
		if (err) {
			printf("Error: Failed to input clEnqueueMigrateMemObjects! %d\n", err);
			exit(EXIT_FAILURE);
		}

		err = clEnqueueTask(q, krnl_integrateKernel, 0, NULL, NULL);
		if (err) {
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(EXIT_FAILURE);
		}


		clFinish(q);

		err = 0;
		err = clEnqueueMigrateMemObjects(q,(cl_uint)1, &buf_in, CL_MIGRATE_MEM_OBJECT_HOST,0,NULL, NULL);
		if (err) {
			printf("Error: Failed to execute kernel! %d\n", err);
			exit(EXIT_FAILURE);
		}

		clFinish(q);
	#endif // INT_NCU

	TOCK("integrateKernel", vol._resolution.x*vol._resolution.y*vol._resolution.z);
}

#ifdef PREF_FUSEVOL
#ifdef FUSE_HP
void prefetchFuseVolInterp (floatH *out, const Volume v, 
	const sMatrix4 pose, const float3 origin, const float3 voxelSize) {
#else
void prefetchFuseVolInterp (float *out, const Volume v, 
	const sMatrix4 pose, const float3 origin, const float3 voxelSize) {
#endif
	
	float3 vsize = v.getSizeInMeters();
	
	unsigned int z, y, x;

	#pragma omp parallel for \
			shared(out), private(z, y, x)
	for (z = 0; z < v._resolution.z; z+=FUSE_LP) {
		int z_idx = z * v._resolution.x*v._resolution.y;

		for (y = 0; y < v._resolution.y; y++) {
			int y_idx = y*v._resolution.x;

			for (x = 0; x < v._resolution.x; x++) {
				uint3 pix = make_uint3(x, y, z);

				float3 pos = v.pos(pix);
				pos = pose * pos;
				
				pos.x += origin.x;
				pos.y += origin.y;
				pos.z += origin.z;
				
				float tsdf;

				if( pos.x < 0 || pos.x >= vsize.x ||
					pos.y < 0 || pos.y >= vsize.y ||
					pos.z < 0 || pos.z >= vsize.z)
				{
					tsdf = 1.0;
				}
				else {
					tsdf = v.interp(pos);
				}

				out[x + y_idx + z_idx] = tsdf;
			}
		}
	}
}
#endif

float4 raycast( const Volume volume,
				const uint2 pos,
				const sMatrix4 view,
				const float nearPlane,
				const float farPlane,
				const float step,
				const float largestep)
{
	const float3 origin = view.get_translation();
	const float3 direction = rotate(view, make_float3(pos.x, pos.y, 1.f));

	const float3 invR = make_float3(1.0f) / direction;
	const float3 tbot = -1 * invR * origin;
	const float3 ttop = invR * (volume.getDimensions() - origin);

	// re-order intersections to find smallest and largest on each axis
	const float3 tmin = fminf(ttop, tbot);
	const float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y),
									 fmaxf(tmin.x, tmin.z));
	const float smallest_tmax = fminf(fminf(tmax.x, tmax.y),
									  fminf(tmax.x, tmax.z));

	// check against near and far plane
	const float tnear = fmaxf(largest_tmin, nearPlane);
	const float tfar = fminf(smallest_tmax, farPlane);

	if (tnear < tfar)
	{
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
		float f_t = volume.interp(origin + direction * t);
		float f_tt = 0;

		if (f_t > 0) // ups, if we were already in it, then don't render anything here
		{
			for (; t < tfar; t += stepsize)
			{
				f_tt = volume.interp(origin + direction * t);
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0)
			{           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return make_float4(origin + direction * t, t);
			}
		}
	}
	return make_float4(0);
}

#ifdef TR_HW
void raycastKernel(float4* vertex, float4* normal, uint2 inputSize,
		const Volume integration, const sMatrix4 view, const float nearPlane,
		const float farPlane, const float step, const float largestep)
#else
void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
		const Volume integration, const sMatrix4 view, const float nearPlane,
		const float farPlane, const float step, const float largestep)
#endif
{
	TICK();
	unsigned int y;
#pragma omp parallel for \
		shared(normal, vertex), private(y)
	#ifdef R_LP
		#ifdef TR_HW
			for (y = 0; y < inputSize.y; y+=2)
				for (unsigned int x = 0; x < inputSize.x; x+=4) {

					uint2 pos = make_uint2(x, y);

					const float4 hit = raycast(integration, pos, view, nearPlane,
							farPlane, step, largestep);
					if (hit.w > 0.0) {
						vertex[pos.x + pos.y * inputSize.x] = hit;
						vertex[pos.x + 1 + pos.y * inputSize.x] = hit;
						vertex[pos.x + 2 + pos.y * inputSize.x] = hit;
						vertex[pos.x + 3 + pos.y * inputSize.x] = hit;
						vertex[pos.x + (pos.y + 1) * inputSize.x] = hit;
						vertex[pos.x + 1 + (pos.y + 1) * inputSize.x] = hit;
						vertex[pos.x + 2 + (pos.y + 1) * inputSize.x] = hit;
						vertex[pos.x + 3 + (pos.y + 1) * inputSize.x] = hit;

						float3 surfNorm = integration.grad(make_float3(hit));
						if (length(surfNorm) == 0) {
							normal[pos.x + pos.y * inputSize.x].x = KFUSION_INVALID;
							normal[pos.x + 1 + pos.y * inputSize.x].x = KFUSION_INVALID;
							normal[pos.x + 2 + pos.y * inputSize.x].x = KFUSION_INVALID;
							normal[pos.x + 3 + pos.y * inputSize.x].x = KFUSION_INVALID;
							normal[pos.x + (pos.y + 1) * inputSize.x].x = KFUSION_INVALID;
							normal[pos.x + 1 + (pos.y + 1) * inputSize.x].x = KFUSION_INVALID;
							normal[pos.x + 2 + (pos.y + 1) * inputSize.x].x = KFUSION_INVALID;
							normal[pos.x + 3 + (pos.y + 1) * inputSize.x].x = KFUSION_INVALID;
						} else {
							normal[pos.x + pos.y * inputSize.x] = make_float4(normalize(surfNorm));
							normal[pos.x + 1 + pos.y * inputSize.x] = make_float4(normalize(surfNorm));
							normal[pos.x + 2 + pos.y * inputSize.x] = make_float4(normalize(surfNorm));
							normal[pos.x + 3 + pos.y * inputSize.x] = make_float4(normalize(surfNorm));
							normal[pos.x + (pos.y + 1) * inputSize.x] = make_float4(normalize(surfNorm));
							normal[pos.x + 1 + (pos.y + 1) * inputSize.x] = make_float4(normalize(surfNorm));
							normal[pos.x + 2 + (pos.y + 1) * inputSize.x] = make_float4(normalize(surfNorm));
							normal[pos.x + 3 + (pos.y + 1) * inputSize.x] = make_float4(normalize(surfNorm));
						}
					} else {
						vertex[pos.x + pos.y * inputSize.x] = make_float4(0);
						vertex[pos.x + 1 + pos.y * inputSize.x] = make_float4(0);
						vertex[pos.x + 2 + pos.y * inputSize.x] = make_float4(0);
						vertex[pos.x + 3 + pos.y * inputSize.x] = make_float4(0);
						vertex[pos.x + (pos.y + 1) * inputSize.x] = make_float4(0);
						vertex[pos.x + 1 + (pos.y + 1) * inputSize.x] = make_float4(0);
						vertex[pos.x + 2 + (pos.y + 1) * inputSize.x] = make_float4(0);
						vertex[pos.x + 3 + (pos.y + 1) * inputSize.x] = make_float4(0);

						normal[pos.x + pos.y * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
						normal[pos.x + 1 + pos.y * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
						normal[pos.x + 2 + pos.y * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
						normal[pos.x + 3 + pos.y * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
						normal[pos.x + (pos.y + 1) * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
						normal[pos.x + 1 + (pos.y + 1) * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
						normal[pos.x + 2 + (pos.y + 1) * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
						normal[pos.x + 3 + (pos.y + 1) * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0, 0);
					}
				}
		#else
			for (y = 0; y < inputSize.y; y+=2)
				for (unsigned int x = 0; x < inputSize.x; x+=4) {

					uint2 pos = make_uint2(x, y);

					const float4 hit = raycast(integration, pos, view, nearPlane,
							farPlane, step, largestep);
					int idx = pos.x + pos.y * inputSize.x;
					if (hit.w > 0.0) {
						vertex[idx] = make_float3(hit);
						vertex[idx + 1] = make_float3(hit);
						vertex[idx + 2] = make_float3(hit);
						vertex[idx + 3] = make_float3(hit);
						vertex[idx + inputSize.x] = make_float3(hit);
						vertex[idx + 1 + inputSize.x] = make_float3(hit);
						vertex[idx + 2 + inputSize.x] = make_float3(hit);
						vertex[idx + 3 + inputSize.x] = make_float3(hit);

						float3 surfNorm = integration.grad(make_float3(hit));
						if (length(surfNorm) == 0) {
							normal[idx].x = KFUSION_INVALID;
							normal[idx + 1].x = KFUSION_INVALID;
							normal[idx + 2].x = KFUSION_INVALID;
							normal[idx + 3].x = KFUSION_INVALID;
							normal[idx + inputSize.x].x = KFUSION_INVALID;
							normal[idx + 1 + inputSize.x].x = KFUSION_INVALID;
							normal[idx + 2 + inputSize.x].x = KFUSION_INVALID;
							normal[idx + 3 + inputSize.x].x = KFUSION_INVALID;
						} else {
							normal[idx] = normalize(surfNorm);
							normal[idx + 1] = normal[idx];
							normal[idx + 2] = normal[idx];
							normal[idx + 3] = normal[idx];
							normal[idx + inputSize.x] = normal[idx];
							normal[idx + 1 + inputSize.x] = normal[idx];
							normal[idx + 2 + inputSize.x] = normal[idx];
							normal[idx + 3 + inputSize.x] = normal[idx];
						}
					} else {
						vertex[idx] = make_float3(0);
						vertex[idx + 1] = make_float3(0);
						vertex[idx + 2] = make_float3(0);
						vertex[idx + 3] = make_float3(0);
						vertex[idx + inputSize.x] = make_float3(0);
						vertex[idx + 1 + inputSize.x] = make_float3(0);
						vertex[idx + 2 + inputSize.x] = make_float3(0);
						vertex[idx + 3 + inputSize.x] = make_float3(0);

						normal[idx] = make_float3(KFUSION_INVALID, 0, 0);
						normal[idx + 1] = make_float3(KFUSION_INVALID, 0, 0);
						normal[idx + 2] = make_float3(KFUSION_INVALID, 0, 0);
						normal[idx + 3] = make_float3(KFUSION_INVALID, 0, 0);
						normal[idx + inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
						normal[idx + 1 + inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
						normal[idx + 2 + inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
						normal[idx + 3 + inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
					}
				}
		#endif
	#else
		for (y = 0; y < inputSize.y; y++)
			for (unsigned int x = 0; x < inputSize.x; x++) {

				uint2 pos = make_uint2(x, y);

				const float4 hit = raycast(integration, pos, view, nearPlane,
						farPlane, step, largestep);
				if (hit.w > 0.0) {
					#ifdef TR_HW
						vertex[pos.x + pos.y * inputSize.x] = hit;	
					#else
						vertex[pos.x + pos.y * inputSize.x] = make_float3(hit);
					#endif
					float3 surfNorm = integration.grad(make_float3(hit));
					if (length(surfNorm) == 0) {
						normal[pos.x + pos.y * inputSize.x].x = KFUSION_INVALID;
					} else {
						#ifdef TR_HW
							normal[pos.x + pos.y * inputSize.x] = make_float4(normalize(surfNorm));
						#else
							normal[pos.x + pos.y * inputSize.x] = normalize(surfNorm);
						#endif
					}
				} else {
					#ifdef TR_HW
						vertex[pos.x + pos.y * inputSize.x] = make_float4(0);
						normal[pos.x + pos.y * inputSize.x] = make_float4(KFUSION_INVALID, 0, 0);
					#else
						vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
						normal[pos.x + pos.y * inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
					#endif
				}
			}
	#endif
	TOCK("raycastKernel", inputSize.x*inputSize.y);
}


bool KFusion::preprocessing(float *inputDepth, const uchar3 *inputRgb)
{
	// downscale resolution
	int ratio = params.compute_size_ratio;

	if (ratio <= 0) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	if (ratio != 1) {
		unsigned int y;
		#pragma omp parallel for private(y)
		for (y = 0; y < params.computationSize.y; y++)
			for (unsigned int x = 0; x < params.computationSize.x; x++) {
				rawDepth[x + params.computationSize.x * y] = inputDepth[x * ratio + params.inputSize.x * y * ratio];
				#ifndef NO_BF
				#ifdef BF_HW
					pad_depth[x+1 + (y+1)*(padSize.x)] = inputDepth[x*ratio + y*ratio*params.inputSize.x];
				#endif
				#endif
			}
		/********* fill rows ************/
		#ifndef NO_BF
		#ifdef BF_HW
			memcpy(pad_depth  + 1, rawDepth, params.computationSize.x*sizeof(float));
			memcpy(pad_depth  + 1 + padSize.x *(padSize.y-1), rawDepth+params.computationSize.x*(params.computationSize.y-1), params.computationSize.x*sizeof(float));
		#endif
		#endif
	}
	else {
		memcpy(rawDepth, inputDepth, params.computationSize.x*params.computationSize.y*sizeof(float));
		
		#ifndef NO_BF
		#ifdef BF_HW
			#pragma omp parallel for 
			for(uint y = 0; y < params.inputSize.y; y++){
				for(uint x = 0; x < params.inputSize.x; x++){
					pad_depth[x+1 + (y+1)*(padSize.x)] = inputDepth[x + y*params.inputSize.x];
				}
			}
			/********* fill rows ************/
			memcpy(pad_depth  + 1, inputDepth, params.inputSize.x*sizeof(float));
			memcpy(pad_depth  + 1 + padSize.x *(padSize.y-1), inputDepth+params.inputSize.x*(params.inputSize.y-1), params.inputSize.x*sizeof(float));
		#endif
		#endif
	}

	#ifndef NO_BF
		#ifdef BF_HW
			/********* fill collumns *********/
			#pragma omp parallel for 
			for(uint y = 0; y <  padSize.y; y++){
				pad_depth[padSize.x*y] = pad_depth[1 + padSize.x*y];
				pad_depth[padSize.x-1 + padSize.x*y] = pad_depth[padSize.x-2 + padSize.x*y];
			}

			bilateralFilterKernel(scaledDepth[0], pad_depth, params.computationSize.x, params.computationSize.y, gaussian, e_delta, radius);
		
		#else
			bilateralFilterKernel(scaledDepth[0], rawDepth, params.computationSize, gaussian, e_delta, radius);	
		#endif
	#else
		memcpy(scaledDepth[0], rawDepth, params.computationSize.x*params.computationSize.y*sizeof(float));
	#endif

	memcpy(rawRgb, inputRgb, params.inputSize.x*params.inputSize.y*sizeof(uchar3));


	return true;
}


bool KFusion::tracking(uint frame)
{
	(void)frame;
	forcePose=false;
	
	// half sample the input depth maps into the pyramid levels
	uint2 localimagesize = params.computationSize;
	for (int i = 1; i < int(iterations.size()); ++i)
	{
		
		halfSampleRobustImageKernel(scaledDepth[i], scaledDepth[i-1], 
									localimagesize, e_delta * 3, 1);
		localimagesize = make_uint2(localimagesize.x >> 1, localimagesize.y >> 1);
	}


	float4 k = make_float4(params.camera.x, params.camera.y, params.camera.z, params.camera.w);
	// prepare the 3D information from the input depth maps
	localimagesize = params.computationSize;
	for (int i = 0; i < int(iterations.size()); ++i)
	{
		depth2vertexKernel(inputVertex[i], scaledDepth[i], localimagesize, 
								getInverseCameraMatrix(k / float(1 << i))); // inverse camera matrix depends on level
		vertex2normalKernel( inputNormal[i], inputVertex[i], localimagesize);

		localimagesize = make_uint2(localimagesize.x >> 1, localimagesize.y >> 1);
	}

	oldPose = pose;
	const sMatrix4 projectReference = camMatrix*inverse(sMatrix4(&raycastPose));
	int i;
	for (int level = iterations.size() - 1; level >= 0; --level)
	{
		uint2 localimagesize = make_uint2(
				params.computationSize.x >> level,
				params.computationSize.y >> level);
		
		for (i = 0; i < iterations[level]; ++i)
		{
			trackPose=pose;
			#ifndef TR_HW
				trackKernel(reduction,
							inputVertex[level],
							inputNormal[level], localimagesize, 
							vertex,
							normal, params.computationSize, 
							pose,
							projectReference,
							dist_threshold,
							normal_threshold);
			#else
				pt_tr_in[0] = inVertex_buffer[level];
				pt_tr_in[1] = inNormal_buffer[level];
				pt_tr_in[2] = refVertex_buffer;
				pt_tr_in[3] = refNormal_buffer;
				pt_tr_in[4] = Ttrack_data_buffer;
				pt_tr_in[5] = view_data_buffer;

				pt_tr_out[0] = trackData_float_buffer;

				TICK();

				int k;
				#pragma omp parallel for
				for (k = 0; k < 4; k ++) {
					Ttrack_data[k*4] = pose.data[k].x;
					Ttrack_data[k*4 + 1] = pose.data[k].y;
					Ttrack_data[k*4 + 2] = pose.data[k].z;
					Ttrack_data[k*4 + 3] = pose.data[k].w;

					view_data[k*4] = projectReference.data[k].x;
					view_data[k*4 + 1] = projectReference.data[k].y;
					view_data[k*4 + 2] = projectReference.data[k].z;
					view_data[k*4 + 3] = projectReference.data[k].w;
				}

				int argcounter = 0;

				err = 0;
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(cl_mem), &trackData_float_buffer);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(cl_mem), &inVertex_buffer[level]);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(cl_mem), &inNormal_buffer[level]);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(int), &localimagesize.x);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(int), &localimagesize.y);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(cl_mem), &refVertex_buffer);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(cl_mem), &refNormal_buffer);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(int), &params.computationSize.x);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(int), &params.computationSize.y);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(cl_mem), &Ttrack_data_buffer);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(cl_mem), &view_data_buffer);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(float), &dist_threshold);
				err |= clSetKernelArg(krnl_trackKernel, argcounter++, sizeof(float), &normal_threshold);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to set krnl_trackKernel arguments! %d\n", err);
			 
				}
				err = clEnqueueMigrateMemObjects(q_tr, (cl_uint)6, pt_tr_in, 0 ,0,NULL, NULL);

				
				err = clEnqueueTask(q_tr, krnl_trackKernel, 0, NULL, NULL);
				if (err) {
					printf("Error: Failed to execute track kernel! %d\n", err);
					return EXIT_FAILURE;
				}	

				err = 0;
				err |= clEnqueueMigrateMemObjects(q_tr, (cl_uint)1, pt_tr_out, CL_MIGRATE_MEM_OBJECT_HOST,0,NULL, NULL);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to write to source array: %d!\n", err);
					return EXIT_FAILURE;
				}

				clFinish(q_tr);

				TOCK("trackKernel", localimagesize.x * localimagesize.y);
			#endif

			reduceKernel(output, reduction, params.computationSize, localimagesize); // compute the linear system to solve
			
			if (updatePoseKernel(pose, output, params.icp_threshold, this->deltaPose))
				break;
		}
	}

	return checkPoseKernel(pose, oldPose, output, params.computationSize, track_threshold);
}

bool KFusion::initKeyFrame(uint frame)
{
	if(frame > 0) {
		VolumeCpu v;
		v.frame = lastKeyFrameIdx;
		v.pose = lastKeyFramePose;
		v.resolution = keyFrameVol.getResolution();
		v.dimensions = keyFrameVol.getDimensions();
		
		uint size = v.resolution.x*v.resolution.y*v.resolution.z;

		#ifndef CHANGE_VOLS
			v.data = new short2[size];

			if(v.data == nullptr)
			{
				std::cerr << "Error allocating memory." << std::endl;
				exit(1);
			}

			memcpy(v.data, keyFrameVol.getDataPtr(), size*sizeof(short2));
		#endif
		volumes.push_back(v);
	}
	
	lastKeyFrameIdx = frame;
	lastKeyFramePose = getPose();
	
	#ifndef CHANGE_VOLS
		initVolumeKernel(keyFrameVol);
	#endif
	lastKeyFrame = frame;

	return true;
}

bool KFusion::fuseVolumes()
{
	//clear volume first    
	#ifndef NO_1ST_INITVOL
		integrate_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,volume._resolution.x*volume._resolution.y*volume._resolution.z*sizeof(short2),0,nullptr,nullptr,&err);
		initVolumeKernel(volume);

		err = clEnqueueUnmapMemObject(q,integrate_vol_buffer,integrate_vol_ptr,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory integrate_vol_buffer!\n");
		}
		clFinish(q);
	#endif

	uint size = fusionVol._resolution.x*fusionVol._resolution.y*fusionVol._resolution.z;
	
	#ifndef FUSE_ZOFF
		int zoff = 0;
	#endif

	#ifdef FUSE_ALLINONE_5

		#ifdef FUSE_HP
		floatH maxweight_h = maxweight;
		#endif
		// kernel invocation
		TICK();
		int step = FUSE_LP;

		int argcounter = 0;

		#if FUSE_NCU != 1
			for (int i = 0; i < FUSE_NCU; i++) {
				pt_fv[i][0] = tsdf_interp_sbuffer[0][i];
				pt_fv[i][1] = tsdf_interp_sbuffer[1][i];
				pt_fv[i][2] = tsdf_interp_sbuffer[2][i];
				pt_fv[i][3] = tsdf_interp_sbuffer[3][i];
				pt_fv[i][4] = tsdf_interp_sbuffer[4][i];
				pt_fv[i][5] = volResolution_buffer;
				pt_fv[i][6] = fuse_vol_sbuffer[i];
			}

			cl_event buffDone[CU], krnlDone[CU], flagDone[CU];
			cl_event event;

			for (int i = 0; i < FUSE_NCU; i++) {
				argcounter = 0;

				// Set arguments 
				err = 0;
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &fuse_vol_sbuffer[i]);
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &fuse_vol_sbuffer[i]);
				#ifdef FUSE_UNROLL_APPROX
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &fuse_vol_sbuffer[i]);
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &fuse_vol_sbuffer[i]);
				#endif
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[0][i]);
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[1][i]);
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[2][i]);
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[3][i]);
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[4][i]);
				#ifdef FUSE_HP
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(floatH), &maxweight_h);
				#else
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(float), &maxweight);
				#endif
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &volResolution_buffer);
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(int), &step);
				err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(int), &zoff);

				if (err != CL_SUCCESS) {
					printf("Error: Failed to set krnl_fuseVolumesKernel[%d] arguments! %d\n", i, err);
			 
				}
				err = clEnqueueMigrateMemObjects(q_fv, (cl_uint)6, pt_fv[i], 0 ,0,NULL, NULL);
				
				err = clEnqueueMigrateMemObjects(q_fv, (cl_uint)1, &pt_fv[i][6], 0 ,0,NULL, &event);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to input integrate_vol_sbuffer[%d]! %d\n", i, err);
					exit(EXIT_FAILURE);
				}
				buffDone[i] = event;
			}

			for (int i = 0; i < FUSE_NCU; i++) {
				err = clEnqueueTask(q_fv, krnl_fuseVolumesKernel[i], 1, &buffDone[i], &krnlDone[i]);
				if (err) {
					printf("Error: Failed to execute kernel krnl_fuseVolumesKernel[%d]! %d\n", i, err);
					exit(EXIT_FAILURE);
				}
			}

			for (int i = 0; i < FUSE_NCU; i++) {
				err = 0;
				err |= clEnqueueMigrateMemObjects(q_fv, (cl_uint)1, &pt_fv[i][6], CL_MIGRATE_MEM_OBJECT_HOST, 1, &krnlDone[i], &flagDone[i]);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to write to source array: %d!\n", err);
					exit(EXIT_FAILURE);
				}
			}

			clWaitForEvents(FUSE_NCU, flagDone);
			
		#else // FUSE_NCU = 1
			pt_fv[0] = tsdf_interp_buffer[0];
			pt_fv[1] = tsdf_interp_buffer[1];
			pt_fv[2] = tsdf_interp_buffer[2];
			pt_fv[3] = tsdf_interp_buffer[3];
			pt_fv[4] = tsdf_interp_buffer[4];
			pt_fv[5] = volResolution_buffer;
			pt_fv[6] = integrate_vol_buffer;
			
			// Set arguments 
			err = 0;
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
			#ifdef FUSE_UNROLL_APPROX
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
			#endif
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[0]);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[1]);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[2]);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[3]);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[4]);
			#ifdef FUSE_HP
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(floatH), &maxweight_h);
			#else
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(float), &maxweight);
			#endif
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &volResolution_buffer);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(int), &step);
			err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(int), &zoff);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to set krnl_fuseVolumesKernel arguments! %d\n", err);
		 
			}
			err = clEnqueueMigrateMemObjects(q_fv, (cl_uint)7, pt_fv, 0 ,0,NULL, NULL);

			
			err = clEnqueueTask(q_fv, krnl_fuseVolumesKernel, 0, NULL, NULL);
			if (err) {
				printf("Error: Failed to execute kernel! %d\n", err);
				exit(EXIT_FAILURE);
			}

			err = 0;
			err |= clEnqueueMigrateMemObjects(q_fv, (cl_uint)1, &pt_fv[6], CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error: Failed to write to source array: %d!\n", err);
				exit(EXIT_FAILURE);
			}

			clFinish(q_fv);
		#endif
		TOCK("fuseVolumesKernel", size);
	#else // FUSE_ALLINONE_5
		#ifdef FUSE_ALLINONE_3
			#if FUSE_NCU != 1
				for (int i = 0; i < FUSE_NCU; i++) {
					pt_fv[i][0] = tsdf_interp_sbuffer[0][i];
					pt_fv[i][1] = tsdf_interp_sbuffer[1][i];
					pt_fv[i][2] = tsdf_interp_sbuffer[2][i];
					pt_fv[i][3] = volResolution_buffer;
					pt_fv[i][4] = integrate_vol_sbuffer[i];
				}

				cl_event buffDone[CU], krnlDone[CU], flagDone[CU];
				cl_event event;

				for (int i = 0; i < FUSE_NCU; i++) {
					int argcounter = 0;

					// Set arguments 
					err = 0;
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
					#ifdef FUSE_UNROLL_APPROX
						err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
						err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &integrate_vol_sbuffer[i]);
					#endif
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[0][i]);
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[1][i]);
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &tsdf_interp_sbuffer[2][i]);
					#ifdef FUSE_HP
						err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(floatH), &maxweight_h);
					#else
						err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(float), &maxweight);
					#endif
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(cl_mem), &volResolution_buffer);
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(int), &step);
					err |= clSetKernelArg(krnl_fuseVolumesKernel[i], argcounter++, sizeof(int), &zoff);

					if (err != CL_SUCCESS) {
						printf("Error: Failed to set krnl_fuseVolumesKernel[%d] arguments! %d\n", i, err);
				 
					}
					err = clEnqueueMigrateMemObjects(q_fv, (cl_uint)4, pt_fv[i], 0 ,0,NULL, NULL);
					
					err = clEnqueueMigrateMemObjects(q_fv, (cl_uint)1, &pt_fv[i][4], 0 ,0,NULL, &event);
					if (err != CL_SUCCESS) {
						printf("Error: Failed to input integrate_vol_sbuffer[%d]! %d\n", i, err);
						exit(EXIT_FAILURE);
					}
					buffDone[i] = event;
				}

				for (int i = 0; i < FUSE_NCU; i++) {
					err = clEnqueueTask(q_fv, krnl_fuseVolumesKernel[i], 1, &buffDone[i], &krnlDone[i]);
					if (err) {
						printf("Error: Failed to execute kernel krnl_fuseVolumesKernel[%d]! %d\n", i, err);
						exit(EXIT_FAILURE);
					}
				}

				for (int i = 0; i < FUSE_NCU; i++) {
					err = 0;
					err |= clEnqueueMigrateMemObjects(q_fv, (cl_uint)1, &pt_fv[i][4], CL_MIGRATE_MEM_OBJECT_HOST, 1, &krnlDone[i], &flagDone[i]);
					if (err != CL_SUCCESS) {
						printf("Error: Failed to write to source array: %d!\n", err);
						exit(EXIT_FAILURE);
					}
				}

				clWaitForEvents(FUSE_NCU, flagDone);
			
			#else // NOT FUSE_NCU
				pt_fv[0] = tsdf_interp_buffer[0];
				pt_fv[1] = tsdf_interp_buffer[1];
				pt_fv[2] = tsdf_interp_buffer[2];
				pt_fv[3] = volResolution_buffer;
				pt_fv[4] = integrate_vol_buffer;

				#ifdef FUSE_HP
				floatH maxweight_h = maxweight;
				#endif
				// kernel invocation
				TICK();
				int step = FUSE_LP;

				int argcounter = 0;
				// Set arguments 
				err = 0;
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[0]);
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[1]);
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[2]);
				#ifdef FUSE_HP
					err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(floatH), &maxweight_h);
				#else
					err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(float), &maxweight);
				#endif
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &volResolution_buffer);
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(int), &step);
				err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(int), &zoff);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to set krnl_fuseVolumesKernel arguments! %d\n", err);
			 
				}
				err = clEnqueueMigrateMemObjects(q_fv, (cl_uint)5, pt_fv, 0 ,0,NULL, NULL);

				
				err = clEnqueueTask(q_fv, krnl_fuseVolumesKernel, 0, NULL, NULL);
				if (err) {
					printf("Error: Failed to execute kernel! %d\n", err);
					exit(EXIT_FAILURE);
				}

				err = 0;
				err |= clEnqueueMigrateMemObjects(q_fv, (cl_uint)1, &pt_fv[4], CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL);
				if (err != CL_SUCCESS) {
					printf("Error: Failed to write to source array: %d!\n", err);
					exit(EXIT_FAILURE);
				}

				clFinish(q_fv);
				TOCK("fuseVolumesKernel", size);
			#endif // FUSE_NCU
		#else  // FUSE_ALLINONE_3
		for(int i=0; i < int(volumes.size()); i++)
		{ 
			VolumeCpu &v=volumes[i];  

			#ifndef PREF_FUSEVOL
				memcpy(fusionVol.data, volumes[i].data, size*sizeof(short2));
			#endif

			#ifndef FUSE_HW
				integrate_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,volume._resolution.x*volume._resolution.y*volume._resolution.z*sizeof(short2),0,nullptr,nullptr,&err);
			#endif

			#ifdef PREF_FUSEVOL
				#ifdef FUSE_HW
					pt_fv[i][0] = tsdf_interp_buffer[i];
					pt_fv[i][1] = volResolution_buffer;
					pt_fv[i][2] = integrate_vol_buffer;

					#ifdef FUSE_HP
					floatH maxweight_h = maxweight;
					#endif
					// kernel invocation
					TICK();
					int step = FUSE_LP;

					int argcounter = 0;
					// Set arguments 
					err = 0;
					err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
					err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &integrate_vol_buffer);
					err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &tsdf_interp_buffer[i]);
					#ifdef FUSE_HP
						err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(floatH), &maxweight_h);
					#else
						err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(float), &maxweight);
					#endif
					err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(cl_mem), &volResolution_buffer);
					err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(int), &step);
					#ifdef FUSE_ZOFF
						err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(int), &i);
					#else
						err |= clSetKernelArg(krnl_fuseVolumesKernel, argcounter++, sizeof(int), &zoff);
					#endif
					if (err != CL_SUCCESS) {
						printf("Error: Failed to set krnl_fuseVolumesKernel arguments! %d\n", err);
				 
					}
					err = clEnqueueMigrateMemObjects(q_fv, (cl_uint)3, pt_fv[i], 0 ,0,NULL, NULL);

					
					err = clEnqueueTask(q_fv, krnl_fuseVolumesKernel, 0, NULL, NULL);
					if (err) {
						printf("Error: Failed to execute kernel! %d\n", err);
						exit(EXIT_FAILURE);
					}

					err = 0;
					err |= clEnqueueMigrateMemObjects(q_fv, (cl_uint)1, &pt_fv[i][2], CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, NULL);
					if (err != CL_SUCCESS) {
						printf("Error: Failed to write to source array: %d!\n", err);
						exit(EXIT_FAILURE);
					}

					clFinish(q_fv);
					TOCK("fuseVolumesKernel", size);

				#else
					fuseVolumesKernelPrefetched(  volume,
										fuseVol_interp_ptr[i],
										inverse(v.pose),
										params.volume_direction,
										maxweight);
				#endif
			#else
				fuseVolumesKernel(  volume,
									fusionVol,
									inverse(v.pose),
									params.volume_direction,
									maxweight);
			#endif

			#ifndef FUSE_HW
				err = clEnqueueUnmapMemObject(q,integrate_vol_buffer,integrate_vol_ptr,0,NULL,NULL);
				if(err != CL_SUCCESS){
					printf("Error: Failed to unmap device memory integrate_vol_buffer!\n");
				}
				clFinish(q);
			#endif
		}
		#endif // FUSE_ALLINONE_3
	#endif // FUSE_ALLINONE_5

	
	#ifdef G2O_OPT_MAX_NUM_KFS
	for (int f_id = 0; f_id < lastKeyFrame; f_id += params.keyframe_rate) {
		dropKeyFrame(f_id);
	}
	#endif
	
	return true;
}

bool KFusion::fuseLastKeyFrame(sMatrix4 &pose)
{
	std::cout << "Fusing last volume" << std::endl;

	#ifndef NO_2ND_INITVOL
		integrate_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,volume._resolution.x*volume._resolution.y*volume._resolution.z*sizeof(short2),0,nullptr,nullptr,&err);
		initVolumeKernel(volume);

		err = clEnqueueUnmapMemObject(q,integrate_vol_buffer,integrate_vol_ptr,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory integrate_vol_buffer!\n");
		}
		clFinish(q);
	#endif

	lastKeyFramePose = pose;
	integrate_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,volume._resolution.x*volume._resolution.y*volume._resolution.z*sizeof(short2),0,nullptr,nullptr,&err);

	fuseVolumesKernel(  volume,
						keyFrameVol,
						inverse(lastKeyFramePose),
						params.volume_direction,
						maxweight);
	err = clEnqueueUnmapMemObject(q,integrate_vol_buffer,integrate_vol_ptr,0,NULL,NULL);
	if(err != CL_SUCCESS){
		printf("Error: Failed to unmap device memory integrate_vol_buffer!\n");
	}
	clFinish(q);

	return true;
}

bool KFusion::raycasting(uint frame)
{
	if (frame > 2) {
		raycastPose = pose;
		integrate_vol_ptr = (short2 *)clEnqueueMapBuffer(q,integrate_vol_buffer,true,CL_MAP_READ | CL_MAP_WRITE,0,volume._resolution.x*volume._resolution.y*volume._resolution.z*sizeof(short2),0,nullptr,nullptr,&err);
				
		#ifdef R_STEP
		raycastKernel(vertex, normal, params.computationSize, volume, 
						raycastPose * inverseCam,
						nearPlane,
						farPlane,
						step,
						0.1125);
		#else
		raycastKernel(vertex, normal, params.computationSize, volume, 
						raycastPose * inverseCam,
						nearPlane,
						farPlane,
						step,
						largestep);
		#endif
		err = clEnqueueUnmapMemObject(q,integrate_vol_buffer,integrate_vol_ptr,0,NULL,NULL);
		if(err != CL_SUCCESS){
			printf("Error: Failed to unmap device memory integrate_vol_buffer!\n");
		}
		clFinish(q);
	}
	else
	{
		return false;
	}

	return true;
}

void KFusion::integrateKeyFrameData()
{
	sMatrix4 delta=inverse(lastKeyFramePose)*pose;
		
	delta(0,3) += params.volume_direction.x;
	delta(1,3) += params.volume_direction.y;
	delta(2,3) += params.volume_direction.z;
	
	integrateKernel(keyFrameVol, rawDepth, rawRgb, params.computationSize, 
										 inverse(delta), camMatrix, params.mu, maxweight, params.compute_size_ratio);
	clFlush(q);

}

bool KFusion::integration(uint frame)
{
	//bool doIntegrate = checkPoseKernel(pose, oldPose, output.data(),params.computationSize, track_threshold);
	if (_tracked || _frame <= 3) {
		
		integrateHWKernel(volume,
						rawDepth,
						rawRgb,
						params.computationSize,
						inverse(pose),
						camMatrix,
						params.mu,
						maxweight, 
						params.compute_size_ratio);
		clFlush(q);
		return true;
	}

	return false;
}

bool KFusion::updatePoseKernel(sMatrix4 & pose, const float * output,float icp_threshold, sMatrix4 &deltaPose)
{
	// Update the pose regarding the tracking result
	TICK();
	bool res = false;

	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	sMatrix4 deltaMat = tosMatrix4(delta);
	sMatrix4 delta4 = deltaMat * sMatrix4(&pose);

	pose.data[0].x = delta4.data[0].x;
	pose.data[0].y = delta4.data[0].y;
	pose.data[0].z = delta4.data[0].z;
	pose.data[0].w = delta4.data[0].w;
	pose.data[1].x = delta4.data[1].x;
	pose.data[1].y = delta4.data[1].y;
	pose.data[1].z = delta4.data[1].z;
	pose.data[1].w = delta4.data[1].w;
	pose.data[2].x = delta4.data[2].x;
	pose.data[2].y = delta4.data[2].y;
	pose.data[2].z = delta4.data[2].z;
	pose.data[2].w = delta4.data[2].w;
	pose.data[3].x = delta4.data[3].x;
	pose.data[3].y = delta4.data[3].y;
	pose.data[3].z = delta4.data[3].z;
	pose.data[3].w = delta4.data[3].w;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold)
	{
		deltaPose=deltaMat;
		res = true;
	}
	
	TOCK("updatePoseKernel", 1);

	return res;
}

bool KFusion::checkPoseKernel(sMatrix4 & pose,
					 sMatrix4 oldPose,
					 const float * output,
					 uint2 imageSize,
					 float track_threshold)
{
	if(forcePose)
	{
		_tracked=true;
		return true;
	}
	
	// Check the tracking result, and go back to the previous camera position if necessary
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ( (std::sqrt(values(0, 0) / values(0, 28)) > 2e-2) ||
		 (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold) )
	{
		pose = oldPose;
		_tracked=false;
		return false;
	}

	_tracked=true;
	return true;
}


