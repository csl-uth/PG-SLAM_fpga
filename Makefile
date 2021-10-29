# Copyright 2020 Xilinx
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make exe HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to build exe application"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."

#trajectories
LR_KT0 := /home/datasets/living_room_traj0_loop_float.raw
LR_KT1 := /home/datasets/living_room_traj1_loop_float.raw
LR_KT2 := /home/datasets/living_room_traj2_loop_float.raw
LR_KT3 := /home/datasets/living_room_traj3_loop_float.raw
RB_TRAJ0 := /home/datasets/rb_traj0.raw
RB_TRAJ1 := /home/datasets/rb_traj1.raw
RB_TRAJ2 := /home/datasets/rb_traj2.raw

ICL_NUIM_PARAMS := 481.2,480,320,240
RB_TRAJ_PARAMS := 570.342,570.342,319.5,239.5

# Points to top directory of Git repository
COMMON_REPO = ./lib/
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

TARGET := hw
HOST_ARCH := aarch64
ifneq ($(NATIVE),yes)
	SYSROOT := /home/fpga
endif
DEVICE := zcu102_base
XCLBIN := xclbin
CONFIG_FILE := design.cfg
BUILD_CPU_CORES = 16
KERNEL_NAME := bilateralFilterKernel


# The C++ Compiler to use is included here, depending arch
include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))


# The below are compile flags are passed to the C++ Compiler
kfusion_FLAGS += -I lib/TooN/include -I lib/include/g2o_slam -I lib/include/multi_voxels -I lib/include  
g2o_FLAGS += -I $(SYSROOT)/usr/include/eigen3 -I $(SYSROOT)/usr/include/suitesparse  -I $(SYSROOT)/home/g2o/EXTERNAL/csparse/ 
system_FLAGS += -I $(SYSROOT)/usr/local/include/  
CXXFLAGS += $(kfusion_FLAGS) $(g2o_FLAGS) $(system_FLAGS) -DG2O_OPT_MAX_NUM_KFS

CXXFLAGS += -Wall -O3 -g -std=c++11 -fopenmp -fmessage-length=0 

ifeq ($(HOST_ARCH),aarch64)
	CXXFLAGS += -march=armv8-a -mtune=cortex-a53 #-mlow-precision-recip-sqrt -ffast-math
endif
# The below are linking flags for C++ Compiler

# Host compiler global settings
LDFLAGS += -lrt -lstdc++ $(opencl_LDFLAGS)
ifneq ($(HOST_ARCH), x86)
ifneq ($(NATIVE),yes)
	LDFLAGS += -Wl,--sysroot=$(SYSROOT)
endif
endif

ifneq ($(NATIVE),yes)
	LDFLAGS += -L $(SYSROOT)/usr/lib/aarch64-linux-gnu -L $(SYSROOT)/usr/local/lib/ 
endif
LDG2O += -lcxsparse -lg2o_cli -lg2o_solver_slam2d_linear -lg2o_types_icp -lg2o_types_slam2d -lg2o_core -lg2o_interface 
LDG2O += -lg2o_solver_csparse -lg2o_solver_structure_only -lg2o_types_sba -lg2o_types_slam3d -lg2o_csparse_extension 
LDG2O += -lg2o_solver_dense -lg2o_parser -lg2o_solver_pcg -lg2o_types_data -lg2o_types_sim3 -lg2o_stuff  
LDOPENCV := -lopencv_calib3d  -lopencv_highgui -lopencv_imgproc -lopencv_xfeatures2d -L $(SYSROOT)/usr/local/lib/ -lopencv_core -lopencv_imgcodecs -lopencv_features2d 


#Host OBJ FILES
HOST_OBJS += lib/include/multi_voxels/utils.o
HOST_OBJS += lib/include/multi_voxels/volume.o
HOST_OBJS += lib/include/multi_voxels/kfusion.o
HOST_OBJS += host_src/g2o_slam3d.o
HOST_OBJS += host_src/main.o


#Host CPP FILES
HOST_CPP_SRCS += host_src/main.cpp
HOST_CPP_SRCS += host_src/g2o_slam3d.cpp
HOST_CPP_SRCS += lib/include/multi_voxels/kfusion.cpp
HOST_CPP_SRCS += lib/include/multi_voxels/utils.cpp
HOST_CPP_SRCS += lib/include/multi_voxels/volume.cpp

# Host Header FILE
HOST_CPP_HDRS += lib/include/g2o_slam/g2o_slam3d.hpp
HOST_CPP_HDRS += lib/include/multi_voxels/kfusion.h
HOST_CPP_HDRS += lib/include/multi_voxels/utils.h
HOST_CPP_HDRS += lib/include/multi_voxels/volume.h


OUTPUT_FILE := benchmark.log
INPUT_FILE := $(RB_TRAJ0)
EXECUTABLE = host
CMD_ARGS +=  -i $(INPUT_FILE) -o $(OUTPUT_FILE) 
CMD_ARGS += -s 8 -p 4.0,4.0,4.0 -r 1 -c 2 -k $(RB_TRAJ_PARAMS)


BINARY_CONTAINERS += $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin
BINARY_CONTAINER_bilateralFilterKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xo

CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig Makefile 

# Building Host
.PHONY: host
host: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_OBJS)
	$(CXX) $(CXXFLAGS) $(HOST_OBJS) -o '$@' $(LDFLAGS) $(LDG2O) $(LDOPENCV) 
	mkdir -p $(XCLBIN)
	$(CP) $(EXECUTABLE) $(XCLBIN)

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $^ -o $@


# Cleaning stuff
RMDIR = rm -rf

clean:
	-$(RMDIR) $(EXECUTABLE) $(HOST_OBJS) 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) host_src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) run.sh 

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin


ECHO := @echo