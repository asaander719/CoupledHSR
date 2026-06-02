import torch
from recbole.model.sequential_recommender.hsr import CoupledHamiltonianBlock
N,T,d,Bv = 4, 50, 16, 4
x = torch.randn(N,T,d); mask = torch.ones(N,T,1); beh = torch.randint(0,Bv,(N,T))
for mode in ['none','symmetric','causal']:
    blk = CoupledHamiltonianBlock(d, 64, kernel_size=4, num_behaviors=Bv, coupling_mode=mode)
    y = blk(x, mask, beh); y.mean().backward()
    assert y.shape==(N,T,d) and torch.isfinite(y).all(), mode
    print(mode, "ok")