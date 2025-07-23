import math
import torch
from transformers import Adafactor

# stochastic rounding for bfloat16
# The implementation was provided by 2kpr. Thank you very much!

def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


# Kahan summation for bfloat16
# The implementation was provided by araleza.
# Base on paper "Revisiting BFloat16 Training": https://arxiv.org/pdf/2010.06192

kahan_residuals = []
tensor_index = 0
prev_step = 0

def copy_kahan_(target: torch.Tensor, source: torch.Tensor, step):
    """
    Copies source into target using Kahan summations.

    The part of the float32 weight that is lost on conversion to bfloat16 is sent
    to the CPU until the next step, where it is re-added onto that step's updated
    weight.  This produces near float32-like weight behavior, although the copies
    back and forth to main memory result in slower training steps.

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    global kahan_residuals, tensor_index, prev_step

    # Calculate the group index of the current residual Tensor. Tensors
    # pass through this copy function in the same order at each step.
    tensor_index += 1
    if prev_step != step:  # Starting new step?
        prev_step = step
        tensor_index = 0

    # Initialize residuals to 0.0 for first step
    if len(kahan_residuals) <= tensor_index:
        kahan_residuals += [torch.zeros_like(source)]

    # Bring the residual from the previous step back from the cpu device, and add it to the
    # float32 weights of the current step
    summed = kahan_residuals[tensor_index].detach().to(source.device)  # Residual is float32 type
    summed.add_(source)

    # Mask off the lower 16 bits of the mantissa, adding 32768 in order to
    # round-to-nearest when the lower bits are clipped off
    summed_i32 = summed.view(dtype=torch.int32).detach().clone()
    summed_quantized_i32 = summed_i32.add_(32768).bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32
    summed_quantized = summed_quantized_i32.view(dtype=torch.float32)

    # The next residual is the difference between the quantized and unquantized weights
    kahan_residuals[tensor_index] = summed.sub(summed_quantized).detach().to("cpu")

    # Copy the quantized floats into the target tensor
    target.copy_(summed_quantized)


@torch.no_grad()
def adafactor_step_param(self, p, group):
    if p.grad is None:
        return
    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("Adafactor does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    factored, use_first_moment = Adafactor._get_options(group, grad_shape)
    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        if use_first_moment:
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(grad)
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
        else:
            state["exp_avg_sq"] = torch.zeros_like(grad)

        state["RMS"] = 0
    else:
        if use_first_moment:
            state["exp_avg"] = state["exp_avg"].to(grad)
        if factored:
            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()

    state["step"] += 1
    state["RMS"] = Adafactor._rms(p_data_fp32)
    lr = Adafactor._get_lr(group, state)

    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
    update = (grad**2) + group["eps"][0]
    if factored:
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]

        exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
        exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

        # Approximation of exponential moving average of square of gradient
        update = Adafactor._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
        update.mul_(grad)
    else:
        exp_avg_sq = state["exp_avg_sq"]

        exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
        update = exp_avg_sq.rsqrt().mul_(grad)

    update.div_((Adafactor._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
    update.mul_(lr)

    if use_first_moment:
        exp_avg = state["exp_avg"]
        exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
        update = exp_avg

    if group["weight_decay"] != 0:
        p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

    p_data_fp32.add_(-update)

    # if p.dtype in {torch.float16, torch.bfloat16}:
    #    p.copy_(p_data_fp32)

    if p.dtype == torch.bfloat16:
        if self.optimizer.use_kahan_summation:
            copy_kahan_(p, p_data_fp32, state["step"])
        else:
            copy_stochastic_(p, p_data_fp32)
    elif p.dtype == torch.float16:
        p.copy_(p_data_fp32)


@torch.no_grad()
def adafactor_step(self, closure=None):
    """
    Performs a single optimization step

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    for group in self.param_groups:
        for p in group["params"]:
            adafactor_step_param(self, p, group)

    return loss


def patch_adafactor_fused(optimizer: Adafactor):
    optimizer.step_param = adafactor_step_param.__get__(optimizer)
    optimizer.step = adafactor_step.__get__(optimizer)
