# Gerneralized Chamfer loss
A generalized Chamfer distance implementation in CUDA for the cases 
where points are unbatchable

## Introduction

Most implementations of Chamfer distance rely on the batchability of tensors.
In this implementation, I removed this constraint and introduce a generalized
Chamfer loss implementation for unbatchable tensors.
Also, this implementation is able to go beyond 3D 
which most existing implementations do not support.
The implementation is written in CUDA with Pytorch front-end, 
and it is fully differentiable so one can easily integrate this loss into
their research.

## Installation

```
git clone https://github.com/justanhduc/stack-chamfer
python setup.py install
```

Binary installation files are available [here](https://yonsei-my.sharepoint.com/:f:/g/personal/adnguyen_o365_yonsei_ac_kr/EnT-GFN4cStLo_dT2JmqCosBcEdCfZB2v9IPyh73p6hwaQ?e=CDHFSa) 
for Pytorch 1.6 and CUDA 10.1 and 10.2.

To test the installation, first install `neuralnet-pytorch` with CUDA extensions
by

```
pip install git+git://github.com/justanhduc/neuralnet-pytorch.git@master --global-option="--cuda-ext"
```

Then execute

```
python -m pytest -sv test.py
```

## Examples

```
import torch
from stack_chamfer import chamfer_loss

xyz11 = torch.rand(2, 4).cuda()  # a point cloud in 4D!!!
xyz11.requires_grad_(True)
xyz12 = torch.rand(4, 4).cuda()
xyz12.requires_grad_(True)
xyz21 = torch.rand(5, 4).cuda()
xyz22 = torch.rand(3, 4).cuda()

xyz1 = torch.cat([xyz11, xyz12])  # unbatchable point cloud
xyz2 = torch.cat([xyz21, xyz22])
batch1 = torch.tensor([0, 0, 1, 1, 1, 1]).cuda()  # create a batch indicator for the stacked point cloud
batch2 = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]).cuda()

loss = chamfer_loss(xyz1, xyz2, batch1, batch2)
```

See [`test.py`](https://github.com/justanhduc/stack-chamfer/blob/master/test.py)
for more examples.

## Known issues

This implementation is not very efficient, as can be seen from the test.
Any PRs regarding this issue is highly welcome.

## Acknowledges

This implementation is highly motivated and inspired by

https://github.com/ThibaultGROUEIX/AtlasNet/

https://github.com/rusty1s/pytorch_cluster

https://github.com/345ishaan/DenseLidarNet
