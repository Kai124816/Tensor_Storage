##Tensor_Storage

Tensor_Storage is a high-performance C++ library designed for efficient storage and manipulation of three dimensional(for now) tensors. It supports various tensor formats and provides parallelized implementations of the Matricized Tensor Times Khatri-Rao Product (MTTKRP) operation on both CPU and AMD GPU architectures.

##Features

Multi-format Tensor Storage: Supports multiple tensor formats, including dense, sparse, and compressed representations.

Parallel MTTKRP: Implements the MTTKRP operation in parallel, leveraging multi-core CPUs and AMD GPUs for accelerated computation.

AMD GPU Support: Utilizes AMD's ROCm platform for GPU acceleration, ensuring compatibility with a wide range of AMD graphics cards.

##Installation Prerequisites

C++ Compiler: Ensure you have a C++17 compatible compiler installed.

ROCm SDK: Required for AMD GPU support. Follow the ROCm installation guide
 to set up the SDK on your system.

 
