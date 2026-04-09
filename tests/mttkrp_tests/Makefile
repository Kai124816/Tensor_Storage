# Compiler settings
CXX = hipcc
CXXFLAGS = -std=c++17 -g -O3 --offload-arch=gfx942 -fopenmp
#Note the --offload-arch=gfx942 flag is specified for the MI300 GPU

# Directories
ROOT_DIR = ../..
GPU_DIR = $(ROOT_DIR)/gpu_code
TENSOR_DIR = $(ROOT_DIR)/tensor_implementations
UTILITY_DIR = $(ROOT_DIR)/utility
BUILD_DIR = build

# Version selection (default: all)
# Options: all, default, in_progress, naive, v1, v2, vectorized, alto
VERSION ?= all
ifdef Version
    VERSION = $(Version)
endif
ifdef version
    VERSION = $(version)
endif

ifeq ("$(VERSION)", "default")
    CXXFLAGS += -DMTTKRP_VERSION_DEFAULT
else ifeq ("$(VERSION)", "in_progress")
    CXXFLAGS += -DMTTKRP_VERSION_IN_PROGRESS
else ifeq ("$(VERSION)", "naive")
    CXXFLAGS += -DMTTKRP_VERSION_NAIVE
else ifeq ("$(VERSION)", "v1")
    CXXFLAGS += -DMTTKRP_VERSION_V1
else ifeq ("$(VERSION)", "v2")
    CXXFLAGS += -DMTTKRP_VERSION_V2
else ifeq ("$(VERSION)", "vectorized")
    CXXFLAGS += -DMTTKRP_VERSION_VECTORIZED
else ifeq ("$(VERSION)", "alto")
    CXXFLAGS += -DMTTKRP_VERSION_ALTO
else
    CXXFLAGS += -DMTTKRP_VERSION_ALL
endif

# Targets
CORRECTNESS_TARGET = correctness_$(VERSION)_bin
TIMING_TARGET = timing_$(VERSION)_bin

# Header dependencies (tracked for rebuilds)
GPU_HEADERS = $(wildcard $(GPU_DIR)/*.h)
TENSOR_HEADERS = $(wildcard $(TENSOR_DIR)/*.h)
UTILITY_HEADERS = $(wildcard $(UTILITY_DIR)/*.h)
ALL_HEADERS = $(GPU_HEADERS) $(TENSOR_HEADERS) $(UTILITY_HEADERS) checksums.h

# GPU kernel source files (compiled separately)
GPU_SRCS = $(GPU_DIR)/vectorized/vectorized_kernels.cc \
           $(GPU_DIR)/one_to_one/one_to_one_kernels.cc \
           $(GPU_DIR)/naive/naive_kernels.cc \
           $(GPU_DIR)/old/old_kernels.cc \
           $(GPU_DIR)/in_progress/in_progress.cc

# GPU object files (in build directory)
GPU_OBJS = $(BUILD_DIR)/vectorized_kernels.o \
           $(BUILD_DIR)/one_to_one_kernels.o \
           $(BUILD_DIR)/naive_kernels.o \
           $(BUILD_DIR)/old_kernels.o \
           $(BUILD_DIR)/in_progress.o

# Test object files (versioned)
CORRECTNESS_OBJ = $(BUILD_DIR)/correctness_$(VERSION).o
TIMING_OBJ = $(BUILD_DIR)/timing_$(VERSION).o

# Default target
all: $(CORRECTNESS_TARGET) $(TIMING_TARGET)

# Alias targets
correctness: $(CORRECTNESS_TARGET)
timing: $(TIMING_TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

#----------------------------------------------------------------------
# GPU kernel compilation (version-independent, compiled once)
#----------------------------------------------------------------------
$(BUILD_DIR)/vectorized_kernels.o: $(GPU_DIR)/vectorized/vectorized_kernels.cc $(ALL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/one_to_one_kernels.o: $(GPU_DIR)/one_to_one/one_to_one_kernels.cc $(ALL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/naive_kernels.o: $(GPU_DIR)/naive/naive_kernels.cc $(ALL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/old_kernels.o: $(GPU_DIR)/old/old_kernels.cc $(ALL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/in_progress.o: $(GPU_DIR)/in_progress/in_progress.cc $(ALL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

#----------------------------------------------------------------------
# Test driver compilation (version-specific, recompiled per VERSION)
#----------------------------------------------------------------------
$(CORRECTNESS_OBJ): correctness.cc $(ALL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TIMING_OBJ): timing.cc $(ALL_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

#----------------------------------------------------------------------
# Link: GPU objects + test driver → final executable
#----------------------------------------------------------------------
$(CORRECTNESS_TARGET): $(CORRECTNESS_OBJ) $(GPU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TIMING_TARGET): $(TIMING_OBJ) $(GPU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: all clean correctness timing

# Cleanup
clean:
	rm -f correctness_* timing_*
	rm -rf $(BUILD_DIR)