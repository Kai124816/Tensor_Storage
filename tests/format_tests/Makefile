# Compiler settings
CXX_1 = g++
CXX_2 = hipcc
CXX_1_FLAGS = -std=c++17 -g -O3 -fopenmp
CXX_2_FLAGS = -std=c++17 -g -O3 --offload-arch=gfx942 -fopenmp
#Note the --offload-arch=gfx942 flag is specified for the MI300 GPU

# Targets
CORRECTNESS_TARGET = correctness
TIMING_TARGET = timing

# Sources
CORRECTNESS_SRC = correctness.cc
TIMING_SRC = timing.cc

# Default target
all: $(CORRECTNESS_TARGET) $(TIMING_TARGET)

# Build rules for g++
$(CORRECTNESS_TARGET): $(CORRECTNESS_SRC)
	$(CXX_1) $(CXX_1_FLAGS) $< -o $@

$(TIMING_TARGET): $(TIMING_SRC)
	$(CXX_1) $(CXX_1_FLAGS) $< -o $@

# Build rules for HIP
hip_correctness: $(CORRECTNESS_SRC)
	$(CXX_2) $(CXX_2_FLAGS) $< -o $(CORRECTNESS_TARGET)

hip_timing: $(TIMING_SRC)
	$(CXX_2) $(CXX_2_FLAGS) $< -o $(TIMING_TARGET)

# Cleanup
clean:
	rm -f $(CORRECTNESS_TARGET) $(TIMING_TARGET)

