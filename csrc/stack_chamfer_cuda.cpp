//
// Created by justanhduc on 20. 9. 16..
//

#include "stack_chamfer_cuda.h"

#include "utils.h"


std::vector<torch::Tensor> stack_chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x,
                                                      at::Tensor idx_y);


std::vector<torch::Tensor> stack_chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x,
                                                       at::Tensor idx_y, at::Tensor idx_xy, at::Tensor idx_yx,
                                                       at::Tensor grad_dist_x, at::Tensor grad_dist_y);


std::vector<torch::Tensor>
stack_chamfer_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x, at::Tensor idx_y) {
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    return stack_chamfer_cuda_forward(xyz1, xyz2, idx_x, idx_y);
}


std::vector<torch::Tensor>
stack_chamfer_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor idx_x, at::Tensor idx_y,
                       at::Tensor idx_xy, at::Tensor idx_yx, at::Tensor grad_dist_x, at::Tensor grad_dist_y) {
    CHECK_INPUT(xyz1);
    CHECK_INPUT(xyz2);
    CHECK_INPUT(grad_dist_x);
    CHECK_INPUT(grad_dist_y);
    return stack_chamfer_cuda_backward(xyz1, xyz2, idx_x, idx_y, idx_xy, idx_yx, grad_dist_x, grad_dist_y);
}
