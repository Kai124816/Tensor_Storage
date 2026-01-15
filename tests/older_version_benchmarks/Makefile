# Compiler settings
CXX      = hipcc
CXXFLAGS = -std=c++17 -g -O3 --offload-arch=gfx942 -fopenmp
#Note the --offload-arch=gfx942 flag is specified for the MI300 GPU

# Targets
BLCO_TARGET = prev_blco_benchmarks

BLCO_SRC = blco_benchmarks.cc

# The first rule is the default (runs when you just type 'make')
all: $(BLCO_TARGET)

# Rule to build ONLY blco
$(BLCO_TARGET): $(BLCO_SRC)
	$(CXX) $(CXXFLAGS) $(BLCO_SRC) -o $(BLCO_TARGET)

# Clean up both
clean:
	rm -f $(BLCO_TARGET)