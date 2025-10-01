## Function Signatures

    __global__ → Marks a function as a GPU kernel (runs on device, launched from host).
    __device__ → Marks a function that runs only on the GPU, callable from other 
    device/global functions.

## Thread indexing variables

    threadIdx.x → thread’s index within its block.
    blockIdx.x → block’s index within the grid.
    blockDim.x → number of threads per block.
    gridDim.x → number of blocks in the grid.
    global_thread_id = blockIdx.x * blockDim.x + threadIdx.x.

## Memory Allocation:

    hipMalloc(void** ptr, size_t size) → Allocates GPU memory.
    hipFree(ptr) → Frees GPU memory.
    hipMemcpy(dst, src, size, kind) → Copies memory between host ↔ device.
    hipMemcpyHostToDevice → CPU → GPU.
    hipMemcpyDeviceToHost → GPU → CPU.

## Kernel Launching:

    hipLaunchKernelGGL(kernel, grid, block, shared_mem, stream, args...) → Launches a kernel.

## General Terms:

    grid = number of blocks.
    block = threads per block.
    wavefront = A group of threads (usually 64) that execute the same instruction in parallel and in lockstep
    shared_mem = dynamic shared memory per block.
    stream = execution stream (usually 0).

## Within Wavefront Operations:

    __shfl(val, srcLane, width) → Warp (wavefront) shuffle: lets threads exchange values without shared memory.
    __ballot(predicate) → Returns bitmask of threads in wavefront that satisfy a predicate.
    __ffsll(mask) → Finds position of first set bit in a 64-bit mask (1-based).
    __popcll(mask) → Population count (number of set bits in 64-bit mask).

## Other Functions:

    atomicAdd(&ptr, value) → Safely adds value to *ptr across threads.
    __builtin_amdgcn_s_barrier() → Full wavefront barrier (synchronizes threads in block).
    hipEventRecord/hipEventSynchronize/hipEventElapsedTime → GPU timing utilities.