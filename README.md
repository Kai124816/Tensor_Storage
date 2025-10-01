## Tensor_Storage

Tensor_Storage is a high-performance C++ library designed for efficient storage and manipulation of three dimensional (for now) tensors. It supports various tensor formats and provides parallelized implementations of the Matricized Tensor Times Khatri-Rao Product (MTTKRP) operation on both CPU and AMD GPU architectures.

## Features

Multi-format Tensor Storage: Supports multiple tensor formats, including dense, sparse, and compressed representations.

Parallel MTTKRP: Implements the MTTKRP operation in parallel, leveraging multi-core CPUs and AMD GPUs for accelerated computation.

AMD GPU Support: Utilizes AMD's ROCm platform for GPU acceleration, ensuring compatibility with a wide range of AMD graphics cards.

## Installation Prerequisites

C++ Compiler: Ensure you have a C++17 compatible compiler installed.

ROCm SDK: Required for AMD GPU support. Follow the ROCm installation guide
 to set up the SDK on your system.

## Downloading Tensors

1. Use wget and the URL of the tensor you want to test. This should return a .gz file
2. Use gunzip to unzip the .gz file.
3. Compile the clean_tns.cc file using the command g++ -std=c++17 -O2 -o clean_tns clean_tns.cc
4. Use the object file to clean out the tensor: (Ex)./clean_tns amazon-reviews.tns 
5. Use the command wc -l tensor.tns to see if the number of non zeros was modified
6. Run your program
7. Remove the tensor file to save space

Note: you can use the head -n 100 to read the first 100 lines of the file and to check what data
type the file is

## Compilation and Execution

Default compilation Instruction to test BLCO tensor: hipcc -std=c++17 -O3 -fopenmp blco_tests.cc -o build/test_blco

other options:
-Wall: For extra warnings
-Wextra: Also for extra warnings
--amdgpu-target=(target gpu) used to compile on target GPU
-DHIP_DEBUG: Debug Macros
-g: Generate debug symbbols which is useful for gdb or hip-gdb
-02: If 03 is too aggressive

Test Tensor Construction (ex): ./test_blco non-zero rows cols depth 
ex: ./test_blco 1000 50 60 70

Test MTTKRP (ex): ./test_blco tensor non-zero rows cols depth mode
ex: ./test_blco tensor_file.txt 1000 50 60 70 2

Test MTTKRP with HIP stats (ex): rocprof --stats -i counters.txt 
./build/test_blco tensors/nell-2.tns 76879419 12092 9184 28818 2 float

## Useful Papers

Useful introduction to tensors: https://www.kolda.net/publication/TensorReview.pdf
Paper on ALTO tensor storage: https://arxiv.org/abs/2102.10245
Paper on BLCO tensor storage: https://arxiv.org/abs/2201.12523



 




