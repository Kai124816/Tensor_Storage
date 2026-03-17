# Compiler settings
CXX      = hipcc
CXXFLAGS = -std=c++17 -g -O3 --offload-arch=gfx942 -fopenmp
#Note the --offload-arch=gfx942 flag is specified for the MI300 GPU

# Targets
BLCO_TARGET = mttkrp_performance

BLCO_SRC = mttkrp_performance.cc

# The first rule is the default (runs when you just type 'make')
all: $(BLCO_TARGET)

$(BLCO_TARGET): $(BLCO_SRC)
	$(CXX) $(CXXFLAGS) $(BLCO_SRC) -o $(BLCO_TARGET)

# Clean up both
clean:
	rm -f $(BLCO_TARGET)