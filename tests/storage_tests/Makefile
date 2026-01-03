# Compiler settings
CXX      = hipcc
CXXFLAGS = -std=c++17 -O3 -fopenmp

# Targets
ALTO_TARGET = test_alto
BLCO_TARGET = test_blco

# Sources
ALTO_SRC = alto_tests.cc
BLCO_SRC = blco_tests.cc

# The first rule is the default (runs when you just type 'make')
all: $(ALTO_TARGET) $(BLCO_TARGET)

# Rule to build ONLY alto
$(ALTO_TARGET): $(ALTO_SRC)
	$(CXX) $(CXXFLAGS) $(ALTO_SRC) -o $(ALTO_TARGET)

# Rule to build ONLY blco
$(BLCO_TARGET): $(BLCO_SRC)
	$(CXX) $(CXXFLAGS) $(BLCO_SRC) -o $(BLCO_TARGET)

# Clean up both
clean:
	rm -f $(ALTO_TARGET) $(BLCO_TARGET)