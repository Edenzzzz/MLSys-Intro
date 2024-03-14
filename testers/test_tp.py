# Launch with torchrun to set tp size (nproc_per_node)
# torchrun --nproc_per_node 4 --master_port 25555 test_tp.py --size 1000
import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing import assert_close

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.modeling_llama import *
from tune_llama import all_reduce_grads, init_dist


class Net(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        rank = dist.get_rank()
        self.weights_0 = nn.Parameter(torch.randn(size, size, device=f"cuda:{rank}", requires_grad=True))
        self.weights_1 = nn.Parameter(torch.randn(size, size, device=f"cuda:{rank}"))
        self.weights_2 = nn.Parameter(torch.randn(size, size, device=f"cuda:{rank}"))
        self.bias_0 = nn.Parameter(torch.randn(size, device=f"cuda:{rank}", requires_grad=True))
        self.bias_1 = nn.Parameter(torch.randn(size, device=f"cuda:{rank}"))
        self.bias_2 = nn.Parameter(torch.randn(size, device=f"cuda:{rank}"))

    def forward(self, x: torch.Tensor):
        out = F.linear(
            F.linear(F.linear(x, self.weights_0, self.bias_0), self.weights_1, self.bias_1),
            self.weights_2,
            self.bias_2,
        )
        return out


def test_tp_correctness(size):
    # Set up distributed environment
    init_dist(dp_size=1)
    batch_size = 4
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    atol = 1e-5
    # There's around 2 outliers due to over/under-flow in TP all-reduce.
    # You could also replace torch.randn with torch.ones to see perfectly accurate results, w/o increasing threshold.
    rtol = 5e-4
    torch.cuda.manual_seed(1)  # Fix randn results

    # Single device forward
    net = Net(size)
    x = torch.randn(batch_size, size, device=f"cuda:{rank}", requires_grad=True)
    out = net(x)
    out.sum().backward()

    # Check TP forward
    tp0 = nn.Linear(size, size).cuda(rank)
    tp0.weight = nn.Parameter(net.weights_0.clone())
    tp0.bias = nn.Parameter(net.bias_0.clone())  # An extra non-parallel linear to check grads propagation
    tp1 = ColumnParallelLinear(net.weights_1, net.bias_1)
    tp2 = RowParallelLinear(net.weights_2, net.bias_2)

    x = tp0(x)
    out_tp = tp2(tp1(x))

    assert out_tp.shape == out.shape
    max_out_err = (out - out_tp).max()
    assert_close(out, out_tp, atol=atol, rtol=rtol) or max_out_err <= 0.03
    print("Forward passed!")

    # Check TP backward
    out_tp.sum().backward()
    all_reduce_grads(tp0)

    # Check scattered gradient
    start = size // world_size * rank
    end = size // world_size * (rank + 1)
    column_split_grad = net.weights_1.grad[start:end]
    row_split_grad = net.weights_2.grad[:, start:end]
    assert_close(column_split_grad, tp1.weight.grad, atol=atol, rtol=rtol)
    assert_close(row_split_grad, tp2.weight.grad, atol=atol, rtol=rtol)
    assert_close(tp0.weight.grad, net.weights_0.grad, atol=atol, rtol=rtol)

    # Check grad for bias
    bias_1_grad = net.bias_1.grad[start:end]
    bias_2_grad = net.bias_2.grad
    assert_close(bias_1_grad, tp1.bias.grad, atol=atol, rtol=rtol)

    # Row Parallel doesn NOT split bias
    if rank == 0:
        assert_close(bias_2_grad, tp2.bias.grad, atol=atol, rtol=rtol)
    assert_close(net.bias_0.grad, tp0.bias.grad, atol=atol, rtol=rtol)
    print("Backward passed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()
    test_tp_correctness(args.size)
