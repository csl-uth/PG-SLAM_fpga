# pg-slam_fpga

This repository contains our MPSoC FPGA configurations for PG-SLAM (Pose Graph - Simultaneous Localization and Mapping) algorithm. PG-SLAM introduces a pose graph optimization module, as an extension on the KinectFusion pipeline, to help recover the humanoid robot's pose. The original CUDA version on ROS of the algorithm is on [this repo](https://github.com/mrsp/slam_stack). Our work implements and evaluates a plethora of MPSoC FPGA designs, exploring a variety of precise and approximate optimizations.

Repository branches:

* sw-only: unoptimized baseline implementation running on ARM with OpenMP
* fastest_precise: fastest and precise HW implementation which implements the Bilateral Filter, Integration and Fuse Voxelgrids kernels on HW and enables all the precise optimization of them
* C1: approximate HW implementation which achieves the best performance (36.04 fps)
* C2: approximate HW implementation which achieves the best performance with zero untracked frames (28.23 fps)
* datasets: includes three trajectories generated in the lab environment using NAO humanoid robot with an Intel Realsense D455 RGB-D camera

## Usage
Vitis™ Unified Software Platform was used to run our experiments on Zynq UltraScale+ MPSoC ZCU102 Evaluation Kit. If you wish to build and run on an Alveo™ card you must make the necessary changes. 

### Requirements for the build environment
1. [Vitis™ Unified Software Platform 2019.2](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis/2019-2.html)
2. [Vitis™ Embedded Platform with DFX 2019.2 or 2020.1](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms/2019-2.html)
3. PetaLinux 2019.2 Sysroot directory for aarch32/aarch64 architectures - [Link](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-design-tools/2019-2.html)
4. Xilinx Runtime (for software emulation on x86 or Alveo™ cards) - [https://github.com/Xilinx/XRT](https://github.com/Xilinx/XRT)

Please consult Xilinx documentation for more detailed information.

### Requirements for the embedded platform
An embedded platform that runs PetaLinux with enabled XRT and OpenCL support. Check this [repository](https://github.com/Xilinx/Vitis_Embedded_Platform_Source) for more information.

### Build
There is a Makefile included that automates the build process for the ZCU102 board. It is based on makefiles provided in [Xilinx Vitis Acceleration Examples](https://github.com/Xilinx/Vitis_Accel_Examples/tree/2019.2). You can make changes to the Makefile according to your setup.

Commands to build for ZCU102:
1. `make bin DEVICE=zcu102_base_dfx`
2. `make host -j4 NATIVE=yes SYSROOT=<path/to/sysroots/aarch64-xilinx-linux>`

The results are created inside `xclbin` directory. Software binary is named `host` and the hardware binary `<name>.hw.xclbin`.

### Run
For our experiments we use the four living-room trajectories [ICL-NUIM dataset](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) and the three trajectories of the datasets branch. We create the `.raw` format using the `scene2raw.cpp` program of the slambench1 repo, using floats to save the pixels. Depending on the input trajectory you must provide the necessary arguments. For instance, to run rb.traj0 (using the default parameters) you must type the command:

`./host -i rb_trj0.raw -s 8 -p 4.0,4.0,4.0 -r 1 -c 2 -k 570.342,570.342,319.5,239.5 -a 50 -e 5 -o output.log -x hwKernels.hw.xclbin` 

To run the ICL-NUIM trajectories, replace camera intrinsics using `-k 481.2,480,320,240`.

### Get in Touch

If you would like to ask questions, report bugs or collaborate on research projects, please email any of the following:

 - Maria Gkeka (margkeka at uth dot gr)
 - Alexandros Patras (patras at uth dot gr)

For more information for our laboratory visit [https://csl.e-ce.uth.gr](https://csl.e-ce.uth.gr) 
