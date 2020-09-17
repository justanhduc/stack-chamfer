//
// Created by justanhduc on 20. 9. 16..
//
#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <ATen/cuda/CUDAContext.h>

#include "utils.h"

#define THREADS 1024


template <typename scalar_t>
__global__ void stack_chamfer_forward_kernel(const scalar_t* x, const scalar_t* y, const int *idx_x, const int *idx_y,
                                             const int dim, scalar_t* dist_yx, int* idx_yx) {
    // dist_xy = min_y ||x-y||^2
    // dist_yx = min_x ||y-x||^2

    const int b = blockIdx.x;
    const int x_start_idx = idx_x[b];
    const int x_end_idx = idx_x[b+1];
    const int y_start_idx = idx_y[b];
    const int y_end_idx = idx_y[b+1];
    
    for(int n_y = y_start_idx + threadIdx.x; n_y < y_end_idx; n_y += THREADS) {
        for(int n_x = x_start_idx; n_x < x_end_idx; ++n_x) {
            scalar_t dist = 0;
            for (int d = 0; d < dim; ++d)
                dist += (x[n_x * dim + d] - y[n_y * dim + d]) * (x[n_x * dim + d] - y[n_y * dim + d]);

            if (dist_yx[n_y] > dist) {
                dist_yx[n_y] = dist;
                idx_yx[n_y] = n_x;
            }
        }
    }
}


std::vector<torch::Tensor>
stack_chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x, at::Tensor idx_y) {
    cudaSetDevice((int)xyz1.device().index());
    const auto xyz1_size = xyz1.size(0);
    const auto xyz2_size = xyz2.size(0);

    auto dist1 = torch::ones(xyz1_size, xyz1.type()) * 1e10;
    auto dist2 = torch::ones(xyz2_size, xyz2.type()) * 1e10;
    auto idx1 = torch::zeros(xyz1_size, idx_x.type());
    auto idx2 = torch::zeros(xyz2_size, idx_y.type());

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::Half, xyz1.scalar_type(), "stack_chamfer_cuda_forward", ([&] {
            stack_chamfer_forward_kernel<scalar_t><<<idx_x.size(0) - 1, THREADS>>>(
                xyz1.data<scalar_t>(), xyz2.data<scalar_t>(), idx_x.data<int>(), idx_y.data<int>(),
                xyz1.size(1), dist2.data<scalar_t>(), idx2.data<int>());
            stack_chamfer_forward_kernel<scalar_t><<<idx_x.size(0) - 1, THREADS>>>(
                xyz2.data<scalar_t>(), xyz1.data<scalar_t>(), idx_y.data<int>(), idx_x.data<int>(),
                xyz1.size(1), dist1.data<scalar_t>(), idx1.data<int>());
        })
    );
    THCudaCheck(cudaGetLastError());

    return { dist1, dist2, idx1, idx2 };
}


template <typename scalar_t>
__global__ void stack_chamfer_backward_kernel(const scalar_t* x, const scalar_t* y, const int* idx_x, const int* idx_xy,
                                              const int dim, const scalar_t* grad_dist, scalar_t* grad_x,
                                              scalar_t* grad_y) {
    const int b = blockIdx.x;
    const int x_start_idx = idx_x[b];
    const int x_end_idx = idx_x[b+1];
    for (int n_x = x_start_idx + threadIdx.x; n_x < x_end_idx; n_x += THREADS) {
        const int n_y = idx_xy[n_x];
        scalar_t g = grad_dist[n_x] * 2;
        for (int d = 0; d < dim; ++d) {
            scalar_t offsets = x[n_x * dim + d] - y[n_y * dim + d];
            atomicAdd(&(grad_x[n_x * dim + d]), g * offsets);
            atomicAdd(&(grad_y[n_y * dim + d]), -g * offsets);
        }
    }
}


std::vector<torch::Tensor>
stack_chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x, at::Tensor idx_y, at::Tensor idx_xy,
                            at::Tensor idx_yx, at::Tensor grad_dist_x, at::Tensor grad_dist_y) {
    cudaSetDevice((int)xyz1.device().index());

    auto grad_xyz1 = torch::zeros_like(xyz1);
    auto grad_xyz2 = torch::zeros_like(xyz2);

    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::Half, xyz1.scalar_type(), "stack_chamfer_cuda_backward", ([&] {
            stack_chamfer_backward_kernel<scalar_t><<<idx_x.size(0) - 1, THREADS>>>(
                xyz1.data<scalar_t>(), xyz2.data<scalar_t>(), idx_x.data<int>(), idx_xy.data<int>(),
                xyz1.size(1), grad_dist_x.data<scalar_t>(), grad_xyz1.data<scalar_t>(), grad_xyz2.data<scalar_t>());
            stack_chamfer_backward_kernel<scalar_t><<<idx_x.size(0) - 1, THREADS>>>(
                xyz2.data<scalar_t>(), xyz1.data<scalar_t>(), idx_y.data<int>(), idx_yx.data<int>(),
                xyz1.size(1), grad_dist_y.data<scalar_t>(), grad_xyz2.data<scalar_t>(), grad_xyz1.data<scalar_t>());
        })
    );
    THCudaCheck(cudaGetLastError());

    return {grad_xyz1, grad_xyz2};
}