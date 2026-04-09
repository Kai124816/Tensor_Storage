#pragma once
#include <hip/hip_runtime.h>
#include "../../tensor_implementations/blco_impl.h"
#include "../kernel_utils.h"

// Host launcher declarations
template<typename T, typename S>
void MTTKRP_BLCO_v1(int mode, Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1);

template<typename T, typename S>
void MTTKRP_BLCO_v2(int mode, Blco_Tensor<T,S>& sparse_tensor, std::vector<float>& times, int iter = 1);

// Prevent implicit instantiation in other translation units
extern template void MTTKRP_BLCO_v1<int, uint64_t>(int, Blco_Tensor<int, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v1<float, uint64_t>(int, Blco_Tensor<float, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v1<unsigned long long, uint64_t>(int, Blco_Tensor<unsigned long long, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v1<double, uint64_t>(int, Blco_Tensor<double, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v1<int, __uint128_t>(int, Blco_Tensor<int, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v1<float, __uint128_t>(int, Blco_Tensor<float, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v1<unsigned long long, __uint128_t>(int, Blco_Tensor<unsigned long long, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v1<double, __uint128_t>(int, Blco_Tensor<double, __uint128_t>&, std::vector<float>&, int);

extern template void MTTKRP_BLCO_v2<int, uint64_t>(int, Blco_Tensor<int, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v2<float, uint64_t>(int, Blco_Tensor<float, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v2<unsigned long long, uint64_t>(int, Blco_Tensor<unsigned long long, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v2<double, uint64_t>(int, Blco_Tensor<double, uint64_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v2<int, __uint128_t>(int, Blco_Tensor<int, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v2<float, __uint128_t>(int, Blco_Tensor<float, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v2<unsigned long long, __uint128_t>(int, Blco_Tensor<unsigned long long, __uint128_t>&, std::vector<float>&, int);
extern template void MTTKRP_BLCO_v2<double, __uint128_t>(int, Blco_Tensor<double, __uint128_t>&, std::vector<float>&, int);