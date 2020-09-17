import torch as T
from torch import nn
from torch.autograd import Function
import stack_chamfer._cuda

__all__ = ['chamfer_loss']


class StackChamferFunction(Function):
    """
    Chamfer's distance module @thibaultgroueix
    GPU tensors only
    """

    @staticmethod
    def forward(ctx, x, y, batch_x=None, batch_y=None):
        x = x.view(-1, 1) if x.dim() == 1 else x
        y = y.view(-1, 1) if y.dim() == 1 else y
        x, y = x.contiguous(), y.contiguous()

        idx_x = None
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1

            deg_x = x.new_zeros(batch_size, dtype=T.long)
            deg_x.scatter_add_(0, batch_x, T.ones_like(batch_x))

            idx_x = deg_x.new_zeros(batch_size + 1)
            T.cumsum(deg_x, 0, out=idx_x[1:])
        else:
            deg_x = None

        idx_y= None
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = int(batch_y.max()) + 1

            deg_y = y.new_zeros(batch_size, dtype=T.long)
            deg_y.scatter_add_(0, batch_y, T.ones_like(batch_y))

            idx_y = deg_y.new_zeros(batch_size + 1)
            T.cumsum(deg_y, 0, out=idx_y[1:])
        else:
            deg_y = None

        idx_x, idx_y = idx_x.int(), idx_y.int()
        dist_x, dist_y, idx_xy, idx_yx = stack_chamfer._cuda.stack_chamfer_forward(x, y, idx_x, idx_y)
        ctx.save_for_backward(x, y, idx_x, idx_y, idx_xy, idx_yx)
        return dist_x, dist_y, deg_x, deg_y

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2, grad_deg_x, grad_deg_y):
        x, y, idx_x, idx_y, idx_xy, idx_yx = ctx.saved_tensors
        grad_dist1 = grad_dist1.contiguous()
        grad_dist2 = grad_dist2.contiguous()
        grad_xyz1, grad_xyz2 = stack_chamfer._cuda.stack_chamfer_backward(
            x, y, idx_x, idx_y, idx_xy, idx_yx, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2, None, None


class StackChamferDistance(nn.Module):
    def __init__(self):
        super(StackChamferDistance, self).__init__()

    def forward(self, x, y, batch_x, batch_y):
        return StackChamferFunction.apply(x, y, batch_x, batch_y)


stack_chamfer_distance = StackChamferDistance()


def chamfer_loss(xyz1, xyz2, batch1=None, batch2=None, reduce='mean'):
    """
    Calculates the Chamfer distance between two batches of point clouds.

    :param xyz1:
        a point cloud of shape ``(b, n1, k)`` or ``(bn1, k)``.
    :param xyz2:
        a point cloud of shape (b, n2, k) or (bn2, k).
    :param batch1:
        batch indicator of shape ``(bn1,)`` for each point in `xyz1`.
        If ``None``, all points are assumed to be in the same point cloud.
        For batched point cloud, this ``None`` must be passed.
    :param batch2:
        batch indicator of shape ``(bn2,)`` for each point in `xyz2`.
        If ``None``, all points are assumed to be in the same point cloud.
        For batched point cloud, this ``None`` must be passed.
    :param reduce:
        ``'mean'`` or ``'sum'``. Default: ``'mean'``.
    :return:
        the Chamfer distance between the inputs.
    """
    assert xyz1.ndim == xyz2.ndim, 'two point clouds do not have the same number of dimensions'
    assert len(xyz1.shape) in (2, 3) and len(xyz2.shape) in (2, 3), 'unknown shape of tensors'
    assert xyz1.shape[-1] == xyz2.shape[-1], 'mismatched feature dimension'
    assert reduce in ('mean', 'sum'), 'Unknown reduce method'

    if xyz1.dim() == 3:
        assert batch1 is None and batch2 is None, 'batch indicators must be None when point clouds are 3D tensors'
        assert xyz1.shape[0] == xyz2.shape[0], 'mismatched batch dimension'

        batch_idx = T.arange(xyz1.shape[0], device=xyz1.device)
        batch_idx = batch_idx[:, None]
        batch1 = T.zeros(*xyz1.shape[:-1], device=xyz1.device, dtype=T.long)
        batch1 += batch_idx
        batch1 = batch1.flatten()
        batch2 = T.zeros(*xyz2.shape[:-1], device=xyz2.device, dtype=T.long)
        batch2 += batch_idx
        batch2 = batch2.flatten()

        xyz1 = xyz1.view(-1, 3)
        xyz2 = xyz2.view(-1, 3)

    dist1, dist2, deg1, deg2 = stack_chamfer_distance(xyz1, xyz2, batch1, batch2)
    if batch1 is None:
        loss_2 = T.mean(dist2)
        loss_1 = T.mean(dist1)
    else:
        loss_1 = T.zeros_like(deg1).to(dist1.dtype)
        loss_1 = T.scatter_add(loss_1, 0, batch1, dist1)
        loss_1 = loss_1 / deg1

        loss_2 = T.zeros_like(deg2).to(dist2.dtype)
        loss_2 = T.scatter_add(loss_2, 0, batch2, dist2)
        loss_2 = loss_2 / deg2

        reduce = T.sum if reduce == 'sum' else T.mean
        loss_1 = reduce(loss_1)
        loss_2 = reduce(loss_2)

    return loss_1 + loss_2
