import torch

from mmcv import Registry
MODELS = Registry('models')

from collections.abc import Iterable
from itertools import repeat
from torch.utils.checkpoint import checkpoint

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)

def checkpoint_sequential(functions, step, input, *args, **kwargs):

    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input, *args)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # the last chunk has to be non-volatile
    end = -1
    segment = len(functions) // step
    for start in range(0, step * (segment - 1), step):
        end = start + step - 1
        input = checkpoint(run_function(start, end, functions), input, preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)
