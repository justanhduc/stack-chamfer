import torch as T
from torch import testing
import neuralnet_pytorch as nnt

from stack_chamfer import chamfer_loss


def test_chamfer_distance():
    # batchable point clouds
    xyz1 = T.rand(2, 4, 3).cuda()
    xyz1.requires_grad_(True)
    xyz2 = T.rand(2, 5, 3).cuda()

    loss = chamfer_loss(xyz1, xyz2)
    loss_gt = nnt.chamfer_loss(xyz1, xyz2)
    testing.assert_allclose(loss, loss_gt)

    loss.backward()
    grad = xyz1.grad
    xyz1.grad *= 0  # zero out grad
    loss_gt.backward()
    grad_gt = xyz1.grad
    testing.assert_allclose(grad, grad_gt)

    # unbatchable point clouds
    xyz11 = T.rand(2, 3).cuda()
    xyz11.requires_grad_(True)
    xyz12 = T.rand(4, 3).cuda()
    xyz12.requires_grad_(True)
    xyz21 = T.rand(5, 3).cuda()
    xyz22 = T.rand(3, 3).cuda()
    batch1 = T.tensor([0, 0, 1, 1, 1, 1]).cuda()
    batch2 = T.tensor([0, 0, 0, 0, 0, 1, 1, 1]).cuda()
    loss = chamfer_loss(T.cat([xyz11, xyz12]), T.cat([xyz21, xyz22]), batch1, batch2)
    loss_gt = (nnt.chamfer_loss(xyz11, xyz21) + nnt.chamfer_loss(xyz12, xyz22)) / 2.
    testing.assert_allclose(loss, loss_gt)

    loss.backward()
    grad11 = xyz11.grad
    grad12 = xyz12.grad
    xyz11.grad *= 0  # zero out grad
    xyz12.grad *= 0  # zero out grad
    loss_gt.backward()
    grad11_gt = xyz11.grad
    grad12_gt = xyz12.grad
    testing.assert_allclose(grad11, grad11_gt)
    testing.assert_allclose(grad12, grad12_gt)

    # beyond 3D
    xyz1 = T.ones(8, 4).cuda()
    batch1 = T.tensor([0, 0, 0, 1, 1, 1, 1, 1]).cuda()
    xyz2 = T.zeros(7, 4).cuda()
    batch2 = T.tensor([0, 0, 0, 0, 1, 1, 1]).cuda()

    loss = chamfer_loss(xyz1, xyz2, batch1, batch2, reduce='sum')
    testing.assert_allclose(loss, 16.)

    # timing
    print('Testing Chamfer distance for batchable case')
    xyz1 = T.rand(64, 1024, 3).cuda()
    xyz2 = T.rand(64, 2048, 3).cuda()
    for _ in range(10):
        print('Stack Chamfer Loss took {}s'.format(nnt.utils.time_cuda_module(chamfer_loss, xyz1, xyz2)))
        print('Old Chamfer Loss took {}s'.format(nnt.utils.time_cuda_module(nnt.chamfer_loss, xyz1, xyz2)))

    def loop_chamfer(xyz1, xyz2):
        loss = sum(nnt.chamfer_loss(xyz1_, xyz2_) for xyz1_, xyz2_ in zip(xyz1, xyz2)) / xyz1.shape[0]
        return loss

    print('Testing Chamfer distance for unbatchable case')
    for _ in range(10):
        print('Stack Chamfer Loss took {}s'.format(nnt.utils.time_cuda_module(chamfer_loss, xyz1, xyz2)))
        print('Old Chamfer Loss took {}s'.format(nnt.utils.time_cuda_module(loop_chamfer, xyz1, xyz2)))
