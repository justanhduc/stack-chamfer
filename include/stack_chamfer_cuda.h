//
// Created by justanhduc on 20. 9. 16..
//

#ifndef EXTENSIONS_STACK_CHAMFER_CUDA_H
#define EXTENSIONS_STACK_CHAMFER_CUDA_H

#include <torch/torch.h>
#include <vector>

std::vector<torch::Tensor> stack_chamfer_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x, at::Tensor idx_y);
std::vector<torch::Tensor> stack_chamfer_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x, at::Tensor idx_y,
                                                  at::Tensor idx_xy, at::Tensor idx_yx, at::Tensor grad_dist_x,
                                                  at::Tensor grad_dist_y);

#endif //EXTENSIONS_STACK_CHAMFER_CUDA_H
