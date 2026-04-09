#pragma once
#include <hip/hip_runtime.h>
#include "../../tensor_implementations/blco_impl.h"
#include "../kernel_utils.h"

// Host launcher declarations
template<typename T, typename S>
void MTTKRP_BLCO_VEC(int mode, Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1);

// Prevent implicit instantiation in other translation units
extern template void MTTKRP_BLCO_VEC<int, uint64_t>(int, Blco_Tensor<int, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_VEC<float, uint64_t>(int, Blco_Tensor<float, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_VEC<unsigned long long, uint64_t>(int, Blco_Tensor<unsigned long long, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_VEC<double, uint64_t>(int, Blco_Tensor<double, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_VEC<int, __uint128_t>(int, Blco_Tensor<int, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_VEC<float, __uint128_t>(int, Blco_Tensor<float, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_VEC<unsigned long long, __uint128_t>(int, Blco_Tensor<unsigned long long, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_VEC<double, __uint128_t>(int, Blco_Tensor<double, __uint128_t>&, std::vector<float>&, int);
