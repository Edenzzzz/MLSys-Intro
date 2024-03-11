# Test LAMB + TP. For starters, we will test on a simple MLP and then Llama convergence
import torch.distributed as dist
import torch.nn as nn

_dim = 500


class net(nn.Module):
    """3 layer MLP for testing LAMB"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(_dim, _dim)
        self.fc2


def main(tp_group: dist.ProcessGroupNCCL = None):
    pass
