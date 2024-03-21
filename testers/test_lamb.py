# Test LAMB + TP. For starters, we will test on a simple MLP and then Llama convergence
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import argparse
from copy import deepcopy

import torchvision
import torchvision.transforms as transforms
import tqdm
from torch.testing import assert_close

from src.dist_utils import init_dist
from src.lamb import COLUMN, ROW, create_lamb_optimizer
from src.modeling_llama import ColumnParallelLinear, RowParallelLinear

_batch_size = 128
_dim = 500
_test_epochs = 4
_lr = 1e-3
_decay = 1e-4
_seed = 1


def assign_grads(model: nn.Module):
    """Used for grad simulation in testing ZeRO2 + TP"""
    for param in model.parameters():
        if param.grad is not None:
            param.grad = torch.randn_like(param)


def load_cifar100():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


def check_params_equal(model, tp_model, group: dist.ProcessGroupNCCL = None, message: str = ""):
    """
    A generic tester for whether params in a TP-adapted model and a
    regular model match.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    for (name, module), (tp_name, tp_module) in zip(model.named_modules(), tp_model.named_modules()):
        if isinstance(tp_module, ColumnParallelLinear):
            # TP split
            split_size = module.weight.size(COLUMN) // world_size
            weight = module.weight.split(split_size, dim=COLUMN)[rank]
            bias = module.bias.split(split_size)[rank]

            assert_close(weight, tp_module.weight)
            assert_close(bias, tp_module.bias)

        elif isinstance(tp_module, RowParallelLinear):
            # TP split
            split_size = module.weight.size(ROW) // world_size
            weight = module.weight.split(split_size, dim=ROW)[rank]

            assert_close(weight, tp_module.weight)
            assert_close(module.bias, tp_module.bias)
        else:
            # Regular module
            if hasattr(module, "weight"):
                assert_close(module.weight, tp_module.weight)
            if hasattr(module, "bias"):
                assert_close(module.bias, tp_module.bias)

    if message != "":
        print(message)


class TestNet(nn.Module):
    def reset_params(self):
        torch.cuda.manual_seed(_seed)
        for name, module in self.named_modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class Net(TestNet):
    """3 layer MLP for testing LAMB"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc0 = nn.Linear(_dim, _dim)
        self.fc1 = nn.Linear(_dim, _dim)
        self.fc2 = nn.Linear(_dim, _dim)

    def forward(self, x):
        x = F.relu(self.fc0(x), inplace=True)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x


class ParallelNet(TestNet):
    """3 layer MLP for testing TP + LAMB"""

    def __init__(self, fc0: nn.Linear, fc1: nn.Linear, fc2: nn.Linear, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        shape = (_dim, _dim)
        self.fc0 = deepcopy(fc0)
        self.fc1 = ColumnParallelLinear(weight=fc1.weight.clone(), bias=fc1.bias.clone(), shape=shape)
        self.fc2 = RowParallelLinear(weight=fc2.weight.clone(), bias=fc2.bias.clone(), shape=shape)
        self.tp_modules = ["fc1", "fc2"]

    def forward(self, x):
        x = F.relu(self.fc0(x), inplace=True)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x


def main(tp_group: dist.ProcessGroupNCCL = None, dp_group: dist.ProcessGroupNCCL = None, test_err: bool = False):
    rank = dist.get_rank(tp_group)

    # Initialize weights to be identical
    torch.manual_seed(_seed)
    model = Net().to(rank)
    tp_model = ParallelNet(fc0=model.fc0, fc1=model.fc1, fc2=model.fc2).to(rank)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, test_loader = load_cifar100()
    total_steps = len(train_loader) * _test_epochs
    check_params_equal(model, tp_model, tp_group, "Initial weights are equal!")

    # Create optimzier and test data
    optim_tp = create_lamb_optimizer(tp_model, _lr, weight_decay=_decay)
    optim = create_lamb_optimizer(model, _lr, weight_decay=_decay)

    # Test optimizer states
    p_bar = tqdm.tqdm(total=total_steps)

    for epoch in _test_epochs:
        for step, (x, y) in enumerate(train_loader):
            out = model(x)
            out_tp = tp_model(x)

            loss = loss_fn(out, y)
            loss_tp = loss_fn(out_tp, y)
            loss.backward()
            loss_tp.backward()

            optim.step()
            optim.zero_grad()
            optim_tp.step()
            optim_tp.zero_grad()

            # check param update
            check_params_equal(model, tp_model, tp_group, message=f"Step {step} passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LAMB + TP")
    parser.add_argument(
        "--assert_err",
        default=True,
        type=eval,
        help="Wheter to assert absolute & relative error, or ",
    )
    parser.add_argument
    args = parser.parse_args()
    init_dist()
    main(test_err=args.test_err)
