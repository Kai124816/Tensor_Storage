## Tensor_Storage

Tensor_Storage is a high-performance C++ library designed for efficient storage and manipulation of three, four, and five dimensional tensors. It supports various tensor formats and provides parallelized implementations of the Matricized Tensor Times Khatri-Rao Product (MTTKRP) operation on both CPU and AMD GPU architectures.

## Features

Multi-format Tensor Storage: Supports multiple tensor formats, including dense, sparse, and compressed representations.

Parallel MTTKRP: Implements the MTTKRP operation in parallel, leveraging multi-core CPUs and AMD GPUs for accelerated computation.

AMD GPU Support: Utilizes AMD's ROCm platform for GPU acceleration, ensuring compatibility with a wide range of AMD graphics cards.

## Installation Prerequisites

C++ Compiler: Ensure you have a C++17 compatible compiler installed.

OpenMP library: Required for Multithreading.

ROCm SDK: Required for AMD GPU support if you want to run the BLCO MTTKRP. Follow the ROCm installation guide to set up the SDK on your system.

## Downloading Tensors

1. Run the download_and_convert.sh file followed by the tensor link, the type of data it holds, and
the number of dimensions. ex() ./download_and_convert.sh https://frostt-tensors.s3.us-east-2.amazonaws.com/1998DARPA/1998darpa.tns.gz int 3
2. The download_and_convert.sh script will generate a tensor.bin file which you can rename if you choose to.

Note: you can use the head -n 100 to read the first 100 lines of the file and to check what data
type the file is

## Compilation and Execution

1. DOWNLOADING TENSORS
----------------------
Tensors can be sourced from the FROSTT repository. Links to these 
tensors are included in 'tensor_list.txt' within the tensors directory. 
After downloading, convert the tensors to the required binary format 
using the provided utility scripts.

2. COMPILATION
--------------
Testing and benchmarking suites are located in the 'tests/' subdirectories.
To compile:
  1. Navigate to the desired subdirectory (e.g., tests/blco).
  2. Run 'make'.

NOTE: The Makefiles default to '--offload-arch=gfx942' for the AMD MI300A. 
If your hardware differs, update the architecture flag in the Makefile 
before running the make command.

3. EXECUTION COMMANDS
---------------------

A. MTTKRP Functional Testing (Correctness)
   Verifies that GPU implementations match CPU Naive results.
   
   Command: ./test_mttkrp <filename> <nnz> <dims...> <mode> <type>
   
   - Use '-none' as the filename to generate a synthetic tensor.
   - To run comprehensive synthetic tests, pass only the rank (3, 4, or 5).
   - Supported Types: int, float, long long, double.

   Example: ./test_mttkrp my_tensor.bin 1000 100 100 100 1 float

B. MTTKRP Benchmarking (Performance)
   Measures execution time and throughput.

   - Kernel Performance: 
     ./kernel <filename> <mode> <nnz> <dims...> <type> <iterations>
   - Memory Allocation Overhead: 
     ./allocation <filename> <nnz> <dims...> <type> <iterations>

C. Legacy BLCO Benchmarks
   Compare against older versions (v1 or v2).
   
   Command: ./benchmark_legacy <v1|v2|all> <file> <mode> <nnz> <d1> <d2> <d3> <type> <iters>

D. Storage Verification
   Tests the integrity of the storage format (ALTO/BLCO).
   
   Command: ./test_storage <filename> <nnz> <dims...> <type>
   - Run with no arguments for a comprehensive storage test suite.

Note: The code for the other test suites is still under development.
Note: If testing on tensor from FROSTT repository lookup the dimensions of the
tensor in tensors/tensor_list.txt

other options to add to makefiles:
-Wall: For extra warnings
-Wextra: Also for extra warnings
--amdgpu-target=(target gpu) used to compile on target GPU
-DHIP_DEBUG: Debug Macros
-g: Generate debug symbbols which is useful for gdb or hip-gdb
-02: If 03 is too aggressive

## Useful Papers

Useful introduction to tensors: https://www.kolda.net/publication/TensorReview.pdf

Paper on ALTO tensor storage: https://arxiv.org/abs/2102.10245

Paper on BLCO tensor storage: https://arxiv.org/abs/2201.12523



 






