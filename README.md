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

Default compilation Instruction to test BLCO tensor: hipcc -std=c++17 -g -O3 --offload-arch=gfx942 -fopenmp. Instructions are contained in MakeFile. (Change --offload-arch= instructions if running on differnt GPU)

other options:
-Wall: For extra warnings
-Wextra: Also for extra warnings
--amdgpu-target=(target gpu) used to compile on target GPU
-DHIP_DEBUG: Debug Macros
-g: Generate debug symbbols which is useful for gdb or hip-gdb
-02: If 03 is too aggressive

Test Tensor Construction: ./test_blco non-zero rows cols depth 
ex: ./test_blco 1000 50 60 70

Test MTTKRP: ./test_blco tensor non-zero rows cols depth mode
ex: ./test_blco ../tensors/darpa.bin 1000 50 60 70 1 int

Test MTTKRP with HIP stats (ex): rocprof --stats -i counters.txt 
./build/test_blco ../tensors/nell-2.bin 76879419 12092 9184 28818 1 float


Default compilation Instruction to test ALTO tensor: hipcc -std=c++17 -O3 -fopenmp alto_tests.cc -o test_alto

other options:
-Wall: For extra warnings
-Wextra: Also for extra warnings
-g: Generate debug symbbols which is useful for gdb or hip-gdb
-02: If 03 is too aggressive

Test Tensor Construction: ./test_blco non-zero rows cols depth 
ex: ./test_alto 1000 50 60 70

Test MTTKRP: ./test_blco tensor non-zero rows cols depth mode
ex: ./test_alto ../tensors/darpa.bin 1000 50 60 70 1 int

Test MTTKRP with HIP stats (ex): rocprof --stats -i counters.txt 
./test_alto ../tensors/nell-2.bin 76879419 12092 9184 28818 1 float

## Useful Papers

Useful introduction to tensors: https://www.kolda.net/publication/TensorReview.pdf

Paper on ALTO tensor storage: https://arxiv.org/abs/2102.10245

Paper on BLCO tensor storage: https://arxiv.org/abs/2201.12523



 






