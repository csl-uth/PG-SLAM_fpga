# PG-SLAM_fpga

This repository contains Xilinx MPSoC FPGA configurations for PG-SLAM (Pose Graph - Simultaneous Localization and Mapping) algorithm. PG-SLAM introduces a pose graph optimization module, as an extension on the KinectFusion pipeline, to help recover the humanoid robot's pose. The original CUDA version on ROS of the algorithm is on [this repo](https://github.com/mrsp/slam_stack). Our work implements and evaluates a plethora of MPSoC FPGA designs, exploring a variety of precise and approximate optimizations.

Repository branches:

* sw-only: unoptimized baseline implementation running on ARM with OpenMP
* fastest_precise: fastest and precise HW implementation which implements the Bilateral Filter, Integration and Fuse Voxelgrids kernels on HW and enables all the precise optimizations of them
* C1: approximate HW implementation which achieves the best performance (36.04 fps)
* C2: approximate HW implementation which achieves the best performance with zero untracked frames (28.23 fps)
* datasets: includes three trajectories generated in the lab environment using NAO humanoid robot with an Intel Realsense D455 RGB-D camera

## Notice
Vitis™ Unified Software Platform was used to run our experiments on Zynq UltraScale+ MPSoC ZCU102 Evaluation Kit. If you wish to build and run on an Alveo™ card you must make the necessary changes. 

PG-SLAM is dependent on OpenCV and g2o libraries and to ease our development process we used the Ubuntu distribution on the ZCU102. The compilation steps for the host code take place on ZCU102 and we do not use any cross-compilation or Petalinux tools for this. Petalinux tools are used only to build the kernel and the boot image.

### Tools for the build environment
1. [Vitis™ Unified Software Platform 2019.2](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2019-2.html)
2. [Vitis™ Embedded Platform 2019.2](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms/2019-2.html)
3. PetaLinux 2019.2 (for kernel and boot image)[Link](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-design-tools/2019-2.html)
4. Xilinx Runtime (for software emulation on x86 or Alveo™ cards) - [https://github.com/Xilinx/XRT](https://github.com/Xilinx/XRT)
5. Ubuntu base Root Filesystem [Link](http://cdimage.ubuntu.com/ubuntu-base/releases/).


### Requirements for the embedded platform 
The operating system that runs on the the embedded platform must have XRT and OpenCL installed. The XRT must be natively compiled on the Ubuntu distribution for the ZCU102. The Xilinx kernel must be built with the Petalinux tools and have the default modules and drivers enabled. Check this [repository](https://github.com/Xilinx/Vitis_Embedded_Platform_Source) for more information on building the platform, the kernel and the boot image. Please consult Xilinx documentation for more detailed information and instructions on how to run Ubuntu OS on the ZCU102.

You will also need to follow these steps:
* Install [OpenCV](https://github.com/opencv/opencv) and [OpenCV_contrib](https://github.com/opencv/opencv_contrib) (version 3.4.14) following the instructions in this repository.

* Install [g2o](https://github.com/RainerKuemmerle/g2o) library: [20200410_git](https://github.com/RainerKuemmerle/g2o/archive/refs/tags/20200410_git.zip) release.

### Build
There is a Makefile included that automates the build process for the ZCU102 board. It is based on makefiles provided in [Xilinx Vitis Acceleration Examples](https://github.com/Xilinx/Vitis_Accel_Examples/tree/2019.2). You can make changes to the Makefile according to your setup.

Commands to build for ZCU102:
1. `make bin DEVICE=zcu102_base` - Execute this command on the development machine with Vitis Compiler installed, to buld the Hardware accelerators.
2. `make host -j4 NATIVE=yes` - Execute this command on the ZCU102, in the Ubuntu OS with necessary libraries installed, to build the host binaries.

The results are created inside `xclbin` directory. Software binary is named `host` and the hardware binary `<name>.hw.xclbin`.

### Run
For our experiments we use the four living-room trajectories [ICL-NUIM dataset](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) and the three trajectories of the datasets branch. We create the `.raw` format using the `scene2raw.cpp` program of the slambench1 repo, but we use floats instead of shorts to save the pixels. Depending on the input trajectory you must provide the necessary arguments. For instance, to run rb.traj0 (using the default parameters) you must type the command:

`./host -i rb_trj0.raw -s 8 -p 4.0,4.0,4.0 -r 1 -c 2 -k 570.342,570.342,319.5,239.5 -a 50 -e 5 -o output.log -x hwKernels.hw.xclbin` 

To run the ICL-NUIM trajectories, update camera intrinsics by using `-k 481.2,480,320,240`.

### Get in Touch

If you would like to ask questions, report bugs or collaborate on research projects, please email any of the following:

 - Maria Gkeka (margkeka at uth dot gr)
 - Alexandros Patras (patras at uth dot gr)

For more information for our laboratory visit [https://csl.e-ce.uth.gr](https://csl.e-ce.uth.gr) 

## License

[MIT](https://choosealicense.com/licenses/mit/)

We note that we use code from [SLAMBench1](https://github.com/pamela-project/slambench1) and [TooN library](https://github.com/edrosten/TooN) repositories, which each one retains its original license.
* slambench1 copyright MIT
* TooN copyright 2-Clause BSD (The original license file also exists on /lib/TooN) 

