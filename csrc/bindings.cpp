#include "stack_chamfer_cuda.h"
#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stack_chamfer_forward", &stack_chamfer_forward, "stack chamfer forward (CUDA)");
    m.def("stack_chamfer_backward", &stack_chamfer_backward, "stack chamfer backward (CUDA)");
}
