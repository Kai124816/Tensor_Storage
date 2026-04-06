# Compiler settings
CXX ?= g++
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall 

# Target executable name
TARGET = get_checksums

# Source files
SRCS = checksums.cc

# Default target
all: $(TARGET)

# Alias target for convenience
checksums: $(TARGET)

# Build rule
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: all clean checksums

# Cleanup
clean:
	rm -f $(TARGET)
