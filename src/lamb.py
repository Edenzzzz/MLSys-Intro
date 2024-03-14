# Disclaimer: Modified from https://github.com/NUS-HPC-AI-Lab/pytorch-lamb/blob/master/optim/lamb.py

from collections.abc import Iterable
from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from .modeling_llama import ColumnParallelLinear, RowParallelLinear

__all__ = ["LAMB"]

# Tensor Parallel split dims
# X @ W^T
COLUMN = 0
ROW = 1


def reduce_scatter_grads(param: Iterable):
    """
    For sharding gradients in ZeRO2. Maintains only a shard of gradients on each device.
    """
    # isintance does NOT guarantee iterable (__getitem__())!
    # (https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable)

    for p in param:
        if p.grad is not None:
            dist.reduce_scatter(p.grad.data, p.grad.data, group=dist.group.WORLD)


def all_gather_states(states: List[torch.Tensor], dim: int = COLUMN, group: dist.ProcessGroup = None):
    """For gathering layer-wise LAMB states in Tensor Parallel."""

    outputs = []
    for tensor in states:
        if not isinstance(tensor, torch.Tensor):
            continue

        shape = tensor.shape
        world_size = dist.get_world_size(group)
        device = dist.get_rank(group)
        output_grad = list(torch.empty(shape, device=device) for _ in range(world_size))

        # TODO: async_op and barrier in the end?
        dist.all_gather(output_grad, tensor, group=dist.group.WORLD)
        outputs.append(torch.cat(output_grad, dim=dim))

        return outputs


class LAMB(Optimizer):
    r"""Implements Lamb(V4) from `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`
    See https://arxiv.org/pdf/1904.00962.pdf
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
        group (dist.ProcessGroup, optional): process group to be used for collective ops
        gather_states (bool, optional): gather states for layer-wise trust ratio computation.
            Default is false; will reduce device-wise norm instead
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        bias_correction=False,
        group: dist.ProcessGroup = None,
        gather_states: bool = False,
    ):
        self.group = group
        self.gather_states = gather_states
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        # if tp:
        #     defaults["is_tp"] = True
        #     defaults["tp_dim"] = tp_dim
        super().__init__(params, defaults)

    def add_tp_params(self, params: Dict, tp_dim: int = COLUMN):
        """
        Add parameters split in Tensor Parallel mode, that will be all-gathered during optimizer.step()
        Arguments:
            params (iterable): As in add_param_group(), dicts defining parameter groups, lr, weight_decay etc.
            tp_dim (int): Dimension along which parameters are scattered across devices
        """
        params["is_tp"] = True
        params["tp_dim"] = tp_dim
        self.add_param_group(params)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        torch.nn.utils.clip_grad_norm_(
            parameters=[p for group in self.param_groups for p in group["params"]], max_norm=1.0, norm_type=2
        )

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients, consider SparseAdam instad.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                scaled_lr = group["lr"]
                if group["bias_correction"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    exp_avg.div_(bias_correction1)
                    exp_avg_sq.div_(bias_correction2)
                update = exp_avg / exp_avg_sq.sqrt().add(group["eps"])

                if group["weight_decay"] != 0:
                    update.add_(p.data, alpha=group["weight_decay"])

                # # All-gather states for layer-wise trust ratio computation
                # if group["is_tp"] and self.gather_states:
                #     p, update = all_gather_states([p, update], dim=group["tp_dim"], group=self.group)
                w_norm = torch.norm(p)
                g_norm = torch.norm(update)

                # To reduce comm costs, just reduce the norms
                is_tp = "tp_dim" in group.keys()
                if is_tp and (not self.gather_states):
                    norms = torch.stack([w_norm, g_norm])
                    dist.all_reduce(norms, group=self.group)
                    w_norm, g_norm = norms.chunk(2)
                trust_ratio = torch.where(w_norm > 0 and g_norm > 0, w_norm / g_norm, torch.ones_like(w_norm))
                scaled_lr *= trust_ratio.item()

                p.data.add_(update, alpha=-scaled_lr)

        return loss


def get_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> List[List[Dict]]:
    """Register decay and no decay param groups for Tensor Parallel,
    for hinting the optimizer whether to communicate the states.
    """
    no_decay_modules = [nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Embedding]
    # Might need to add more modules to make it robust... but others are rarely in transformers?
    tp_modules = [ColumnParallelLinear, RowParallelLinear]  # TODO: add parallel vocab embedding during integration
    decay_modules = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d] + tp_modules

    # Log parallelized dims for general use-cases like all-gather (not necessary in LAMB)
    decay_group = dict(params=[], weight_decay=weight_decay)
    no_decay_group = dict(params=[], weight_decay=0)
    tp_decay_group = {
        COLUMN: dict(params=[], weight_decay=weight_decay, tp_dim=COLUMN),
        ROW: dict(params=[], weight_decay=weight_decay, tp_dim=ROW),
    }
    tp_no_decay_group = {
        COLUMN: dict(params=[], weight_decay=0, tp_dim=COLUMN)
    }  # We do NOT split bn, ln, or bias in row parallel

    for m_name, module in model.named_modules():
        for p_name, param in module.named_parameters():
            if any(isinstance(module, b) for b in no_decay_modules):
                # normalization layers
                no_decay_group["params"].append(param)

            elif any(isinstance(module, w) for w in decay_modules) and p_name.endswith("bias"):
                if isinstance(module, RowParallelLinear):
                    # Each rank holds the full bias; no comm needed
                    no_decay_group["params"].append(param)

                elif isinstance(module, ColumnParallelLinear):
                    # Need to comm states
                    tp_no_decay_group[COLUMN]["params"].append(param)
                else:
                    # Regular non-parallel modules
                    no_decay_group["params"].append(param)

            elif any(isinstance(module, w) for w in decay_modules) and p_name.endswith("weight"):
                # Mostly linear layers in transformer
                if isinstance(module, RowParallelLinear):
                    tp_decay_group[ROW]["params"].append(param)
                elif isinstance(module, ColumnParallelLinear):
                    tp_decay_group[COLUMN]["params"].append(param)
                else:
                    decay_group["params"].append(param)

    # Reformat to directly feed to optimizer
    tp_decay_group = list(tp_decay_group.values())
    tp_no_decay_group = list(tp_no_decay_group.values())
    decay_group = [decay_group]
    no_decay_group = [no_decay_group]
    return decay_group + no_decay_group + tp_decay_group + tp_no_decay_group


def create_lamb_optimizer(
    model,
    lr,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0,
    bias_correction=True,
    group: dist.ProcessGroup = None,
    tp: bool = False,
):
    init_params = get_param_groups(model, weight_decay)
    init_params = [group for group in init_params if len(group["params"]) > 0]
    assert len(init_params) > 0, "No params added to optimizer init. Check your blacklist and get param function! "

    optimizer = LAMB(
        init_params,
        lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        bias_correction=bias_correction,
        group=group,
    )
    return optimizer
