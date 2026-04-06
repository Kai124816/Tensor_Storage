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

A. MTTKRP Testing
  Time MTTKRP code and verify it's correctness.

  - Kernel Correctness:
    Compile Code: Make correctness VERSION=<selected version>
    Run Binary: ./<binary_name> <version> <filename> <nnz> <dims...> <mode> <type>

  - Kernel Performance: 
    Compile Code: Make timing VERSION=<selected version>
    Run Binary: ./kernel <version> <filename> <mode> <nnz> <dims...> <iterations> <type>.

  - Available Versions: all, default(one to one kernels), in_progress, naive, v1, v2, vectorized, alto
  
  Notes:
  - You can pass in VERSION=all to compile a fat binary for all versions
  - Use '-none' as the filename to generate a synthetic tensor.
  - Supported Types: int, float, long long, double.

B. Hardware Profiling (Rocprof)
  Profiles the GPU kernel using rocprofv3 to collect hardware counters.
  This test suite is located in the 'tests/rocprof_tests' directory.

  Step 1: Run the profiler
    Command: ./run_profiler.sh <Tensor Name> <Binary Name> <Version> <Mode> <Counter File>
    Example: ./run_profiler.sh Darpa ./correctness_in_progress 1 basic.yaml

  Step 2: Clean up results and format to CSV
    Command: ./clean_up_results.sh <Tensor Name> <Mode> <Counter .txt File> <Number of CSVs> <Output CSV>
    Example: ./clean_up_results.sh Darpa 1 basic.txt 2 output.csv

  Notes:
  - Same versions available as for MTTKRP testing besides alto since alto is implemented on the CPU.
  - The csv_editor.py file has some different options to help you edit your CSV file.

C. Format Testing
  Time tensor generatio and verify its correctness.

  - Generation Correctness:
    Compile Code: Make correctness
    Run Binary: ./<binary_name> <version> <filename> <nnz> <dims...> <mode> <type>

  - Generation Performance: 
    Compile Code: Make timing
    Run Binary: ./kernel <version> <filename> <mode> <nnz> <dims...> <iterations> 

  Notes:
  - You can also compile the code using the hipcc compiler using the commands make hip_correctness or make hip_timing
  - You can compile both using the command Make all
  - Use '-none' as the filename to generate a synthetic tensor.
  - Supported Types: int, float, long long, double.
    
Notes: 
- The code for the test suites is still under development and may have bugs.
- When compiling with the HIPCC compiler the compiler contains the flag --offload-arch=gfx942 which optimizes the binary for the AMD MI300A GPU. You can change this flag if you want to run the kernel on a different architecture.

## Useful Papers

Useful introduction to tensors: https://www.kolda.net/publication/TensorReview.pdf

Paper on ALTO tensor storage: https://arxiv.org/abs/2102.10245

Paper on BLCO tensor storage: https://arxiv.org/abs/2201.12523



 







