import os

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    x1 = torch.ones(10000, 10000, device=rank)
    handle = dist.all_reduce(x1, op=dist.ReduceOp.SUM, async_op=True)
    # handle.wait() # without this, x1 += 1 might finish before all_reduce in another stream
    # x1 += 1
    print(x1[0][0])


if __name__ == "__main__":
    main()
