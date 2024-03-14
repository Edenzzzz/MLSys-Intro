from torch import nn

__all__ = [
    "get_bn_parameters",
    "get_ln_parameters",
    "get_norm_parameters",
    "get_norm_bias_parameters",
    "get_common_parameters",
]

BN_CLS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_bn_parameters(module):
    return get_parameters_from_cls(module, BN_CLS)


def get_ln_parameters(module):
    return get_parameters_from_cls(module, nn.LayerNorm)


def get_norm_parameters(module):
    return get_parameters_from_cls(module, (nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and "bias" in name:
            yield param


def get_norm_bias_parameters(module):
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_layers=[], include_layers=[]):
    assert not (exclude_layers and include_layers), "You must specify only include some layers or only exclude some"
    exclude = len(exclude_layers) > 0

    names = []
    modules = []
    for layer in exclude_layers:
        if layer == "bias":
            names.append(layer)
        elif layer == "bn":
            modules.extend(BN_CLS)
        elif layer == "ln":
            modules.append(nn.LayerNorm)
        elif isinstance(layer, nn.Module):
            modules.append(layer)

    def get_members_fn(m):
        include_this = exclude ^ isinstance(m, tuple(modules))
        if include_this:
            return m._parameters.items()
        else:
            return dict()

    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        include_this = exclude ^ any([n in name for n in names])
        if include_this:
            # print(name)
            yield param
