# Compiler settings
CXX = hipcc
CXXFLAGS = -std=c++17 -g -O3 --offload-arch=gfx942 -fopenmp
#Note the --offload-arch=gfx942 flag is specified for the MI300 GPU

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
CORRECTNESS_TARGET = correctness_$(VERSION)
TIMING_TARGET = timing_$(VERSION)

# Sources
CORRECTNESS_SRC = correctness.cc
TIMING_SRC = timing.cc

# Default target
all: $(CORRECTNESS_TARGET) $(TIMING_TARGET)

# Alias targets for convenience
correctness: $(CORRECTNESS_TARGET)
timing: $(TIMING_TARGET)

# Build rules
$(CORRECTNESS_TARGET): $(CORRECTNESS_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

$(TIMING_TARGET): $(TIMING_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: all clean correctness timing

# Cleanup
clean:
	rm -f correctness_* timing_* correctness timing