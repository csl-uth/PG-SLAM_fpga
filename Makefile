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
	SYSROOT := /home/fpga_fp
endif
DEVICE := zcu102_base
XCLBIN := xclbin
CONFIG_FILE := design.cfg
BUILD_CPU_CORES = 16
KERNEL_NAME := hwKernels
KERNEL_NAME_1 := bilateralFilterKernel
KERNEL_NAME_3 := integrateKernel
KERNEL_NAME_4 := fuseVolumesKernel


# Enable Profiling
REPORT := no
PROFILE:= no


# The C++ Compiler to use is included here, depending arch
include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))
BUILD_DIR := ./build/build_dir.$(TARGET).$(XSA)
BUILD_DIR_hwKernels = $(BUILD_DIR)/$(KERNEL_NAME)


# The kernel Compiler to use : V++
VPP := v++

#Include Libraries
# Definition of include file locations
xrt_path = $(XILINX_XRT)
ifneq ($(HOST_ARCH), x86)
	xrt_path =  $(SYSROOT)/opt/xilinx/xrt
endif

OPENCL_INCLUDE:= $(xrt_path)/include
ifneq ($(HOST_ARCH), x86)
	OPENCL_INCLUDE:= $(SYSROOT)/opt/xilinx/xrt/include
endif

opencl_CXXFLAGS=-I$(OPENCL_INCLUDE) -I$(VIVADO_INCLUDE)
OPENCL_LIB:= $(xrt_path)/lib
ifneq ($(NATIVE),yes)
## Cross-compile
opencl_LDFLAGS= -L$(OPENCL_LIB) -lOpenCL -lpthread 
VIVADO_INCLUDE:= $(XILINX_VIVADO)/include
else
## Build on FPGA
VIVADO_INCLUDE := /home/vitis_includes/include 
opencl_LDFLAGS= -lOpenCL -lpthread 
endif

# The below are compile flags are passed to the C++ Compiler
kfusion_FLAGS += -I lib/TooN/include -I lib/include/g2o_slam -I lib/include/multi_voxels -I lib/include  
g2o_FLAGS += -I $(SYSROOT)/usr/include/eigen3 -I $(SYSROOT)/usr/include/suitesparse  -I $(SYSROOT)/home/g2o/EXTERNAL/csparse/ 
system_FLAGS += -I $(SYSROOT)/usr/local/include/  
CXXFLAGS += $(kfusion_FLAGS) $(g2o_FLAGS) $(system_FLAGS)

CXXFLAGS += -Wall -O3 -g -std=c++11 -fopenmp -fmessage-length=0 
CXXFLAGS += -DHLS_NO_XIL_FPO_LIB -DINT_HP -DINT_NCU -DINT_LP4 
CXXFLAGS += -DG2O_OPT_MAX_NUM_KFS -DNO_2ND_INITVOL -DNO_1ST_INITVOL 
CXXFLAGS += -DNO_BF -DTR_SW_APPROX
CXXFLAGS += -DINTKF
CXXFLAGS += -DR_LP -DR_RATE -DR_TRINT -DR_STEP -DR_INTERLEAVING 
CXXFLAGS += -DNO_FUSELASTKF -DPREF_FUSEVOL -DFUSE_LP=6 -DFUSE_HW -DCHANGE_VOLS -DFUSE_HP -DFUSE_ALLINONE_5 -DFUSE_NCU=1 

ifeq ($(HOST_ARCH),aarch64)
	CXXFLAGS += -march=armv8-a -mtune=cortex-a53 -mlow-precision-recip-sqrt -ffast-math
endif
# The below are linking flags for C++ Compiler
LDFLAGS += -lrt -lstdc++ $(opencl_LDFLAGS)

# Host compiler global settings
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


######## Hardware Kernel compiler global settings
CLFLAGS_NOLINK += -t $(TARGET) --jobs $(BUILD_CPU_CORES) --platform $(DEVICE) --save-temps $(kfusion_FLAGS) \
--advanced.prop kernel.bilateralFilterKernel.kernel_flags="-std=c++0x -fexceptions" -g \
--advanced.prop kernel.integrateKernel.kernel_flags="-std=c++0x -fexceptions -DHLS_NO_XIL_FPO_LIB " \
--advanced.prop kernel.fuseVolumesKernel.kernel_flags="-std=c++0x -fexceptions" 


CLFLAGS += -t $(TARGET) --jobs $(BUILD_CPU_CORES) --platform $(DEVICE) --config $(CONFIG_FILE) --save-temps $(kfusion_FLAGS) 
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif

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
INPUT_FILE := $(VIPGPU1)
EXECUTABLE = host
CMD_ARGS +=  -i $(INPUT_FILE) -o $(OUTPUT_FILE) -X $(BINARY_CONTAINERS)
CMD_ARGS += -s 8 -p 4.0,4.0,4.0 -r 1 -c 2 -k $(BAG_PARAMS) -a 50 -e 5 
EMCONFIG_DIR = $(XCLBIN)
EMU_DIR = $(SDCARD)/data/emulation


BINARY_CONTAINERS += $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin
BINARY_CONTAINER_bilateralFilterKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME_1).$(TARGET).xo
BINARY_CONTAINER_integrateKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME_3).$(TARGET).xo
BINARY_CONTAINER_fuseVolumesKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME_4).$(TARGET).xo

CP = cp -rf
MV = mv
XCLBINITUTIL = xclbinutil


.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig Makefile 


# Building kernel
.PHONY: bin
bin: $(BINARY_CONTAINERS)

# Building kernel
$(BINARY_CONTAINER_bilateralFilterKernel_OBJS): hls_source/bilateralFilterKernel.cpp
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS_NOLINK) --temp_dir $(BUILD_DIR_hwKernels) -c -k $(KERNEL_NAME_1) -I'$(<D)' -o'$@' '$<'
$(BINARY_CONTAINER_integrateKernel_OBJS): hls_source/integrateKernel.cpp
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS_NOLINK) --temp_dir $(BUILD_DIR_hwKernels) -c -k $(KERNEL_NAME_3) -I'$(<D)' -o'$@' '$<'
$(BINARY_CONTAINER_fuseVolumesKernel_OBJS): hls_source/fuseVolumesKernel.cpp
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS_NOLINK) --temp_dir $(BUILD_DIR_hwKernels) -c -k $(KERNEL_NAME_4) -I'$(<D)' -o'$@' '$<'
$(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin: $(BINARY_CONTAINER_integrateKernel_OBJS) $(BINARY_CONTAINER_fuseVolumesKernel_OBJS) $(BINARY_CONTAINER_bilateralFilterKernel_OBJS)
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR_hwKernels) -l $(LDCLFLAGS) -o'$@' $(+)
	$(XCLBINITUTIL) --force --dump-section BITSTREAM:RAW:$(KERNEL_NAME).$(TARGET).bit -i $(BINARY_CONTAINERS)
	$(MV) $(KERNEL_NAME).$(TARGET).bit $(XCLBIN)

ifneq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	mkdir -p reports
	$(CP) $(BUILD_DIR_hwKernels)/reports/* ./reports/
	$(CP) $(BUILD_DIR_hwKernels)/reports/link/imp/kernel_util_synthed.rpt ./

endif

	
# Building Host
.PHONY: host
host: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_OBJS)
	$(CXX) $(CXXFLAGS) $(HOST_OBJS) -o '$@' $(LDFLAGS) $(LDG2O) $(LDOPENCV) 
	mkdir -p $(XCLBIN)
	$(CP) $(EXECUTABLE) $(XCLBIN)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@


emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: bin exe emconfig
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	export XCL_EMULATION_MODE=$(TARGET) &&	./$(EXECUTABLE) $(CMD_ARGS)
else
	mkdir -p $(EMU_DIR)
	$(CP) $(XILINX_VITIS)/data/emulation/unified $(EMU_DIR)
	mkfatimg $(SDCARD) $(SDCARD).img 500000
	launch_emulator -no-reboot -runtime ocl -t $(TARGET) -sd-card-image $(SDCARD).img -device-family $(DEV_FAM)
endif
else
ifeq ($(HOST_ARCH), x86)
	 ./$(EXECUTABLE) $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin
endif
endif
ifeq ($(HOST_ARCH), x86)
ifeq ($(PROFILE), yes)
	perf_analyze profile -i profile_summary.csv -f html
endif
endif


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

check_xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly and rerun)
endif


ECHO := @echo

#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
CLFLAGS += --report_level estimate
CLLDFLAGS += --report_level system
endif

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all:all
LDCFLAGS += --profile_kernel  stall:all:all:all:all
LDCFALGS += --profile_kernel exec:all:all:all:all
endif
