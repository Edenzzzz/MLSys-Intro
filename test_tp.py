# Launch with torchrun to set tp size (nproc_per_node)
# torchrun --nproc_per_node 4 --master_port 25555 test_tp.py --size 1000
from src.modeling_llama import *
import pytest
import torch
import torch.nn as nn
from tune_llama import init_dist
import argparse
import torch.distributed as dist

def test_tp_correctness(size):
    batch_size = 4
    init_dist(dp_size=1)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    torch.cuda.manual_seed(42)
    weights_1 = nn.Parameter(torch.ones(size, size, device=f"cuda:{rank}"))
    weights_2 = nn.Parameter(torch.ones(size, size, device=f"cuda:{rank}"))
    bias_1 = nn.Parameter(torch.ones(size, device=f"cuda:{rank}"))
    bias_2 = nn.Parameter(torch.ones(size, device=f"cuda:{rank}"))
    
    x = torch.ones(batch_size, size, device=f"cuda:{rank}", requires_grad=True)
    out = F.linear(F.linear(x, weights_1, bias_1), weights_2, bias_2)
    
    # Test TP output
    tp1 = ColumnParallelLinear(weights_1, bias_1)
    tp2 = RowParallelLinear(weights_2, bias_2)
    out_tp = tp2(tp1(x))
    
    assert out_tp.shape == out.shape    
    assert torch.allclose(out, out_tp, atol=1e-5), f"Mean error: {(out - out_tp).mean()}"
    print("Forward passed!")
    
    # Test TP backward
    out_tp.sum().backward()
    out.sum().backward()

    column_split_grad = weights_1.grad[size // world_size * rank: size // world_size * (rank + 1)]
    row_split_grad = weights_2.grad[:, size // world_size * rank: size // world_size * (rank + 1)]
    assert torch.allclose(column_split_grad, tp1.weight.grad, atol=1e-5), f"Mean error: {(column_split_grad - tp1.weight.grad).mean()}"
    assert torch.allclose(row_split_grad, tp2.weight.grad, atol=1e-5), f"Mean error: {(row_split_grad - tp2.weight.grad).mean()}"
    print("Backward passed!")
    
    dist.destroy_process_group() 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()
    test_tp_correctness(args.size)