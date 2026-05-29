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
# Based on paper "Revisiting BFloat16 Training": https://arxiv.org/pdf/2010.06192

def copy_kahan_(target: torch.Tensor, source: torch.Tensor, state, update):
    """
    Copies source into target using Kahan summation.

    The lower bits of the float32 weight that are lost on conversion to bfloat16
    are sent to the CPU until the next step, where they are re-added onto the weights
    before adding the gradient update.  This produces near float32-like weight behavior,
    although the copies back and forth to main memory result in slower training steps.

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
        state:  the optimizer state, used to store kahan residuals
        update: the change in weights due to the gradient
    """

    # Initialize residuals to 0 for first step
    if state.get('kahan_residuals') is None:
        state['kahan_residuals'] = torch.zeros_like(source, dtype=torch.int16)
    
    # Need this in 32 bit as PyTorch doesn't support mixed 32-bit and 16-bit math operations
    state['kahan_residuals'] = state['kahan_residuals'].to(source.device).to(dtype=torch.int32)
    
    # Bring the previous step's lower bits of the weights back from the
    # cpu device, and add them back to the weights of the current step.
    source_i32 = source.view(dtype=torch.int32)  # Can't do math on uint32
    source_i32.add_(state['kahan_residuals'])

    # Reverse any rounding up during the cast to bf16 on the previous step
    rounded_up = state['kahan_residuals'] >= 32768
    source_i32[rounded_up] -= 65536

    # Must add the gradient update after the bottom bits are restored in case
    # the exponent is changed by the update, or the -65536 on the line above
    # would drop the uint32 value below zero, which is invalid.
    source.add_(-update)

    # Get the lower bits into the residual
    torch.bitwise_and(source_i32, 0x0000FFFF, out=state['kahan_residuals'])

    # Ensure rounding to bfloat16 matches expectations. These lines may not be
    # necessary as target.copy_ should do this rounding anyway.
    source_i32.add_(32768)  # Add offset so clipping bits performs round-to-nearest
    source_i32.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32. Leaves only upper bits in source

    # Move the 16-bit Kahan bits from VRAM to main memory
    state['kahan_residuals'] = state['kahan_residuals'].to(dtype=torch.uint16).to("cpu")

    # Copy the quantized floats into the target tensor
    target.copy_(source)


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

    # Add on gradient update, but not if using kahan summation as the bottom
    # bits must be restored first. (This update occurs in copy_kahan_() instead)
    if not self.optimizer.use_kahan_summation:
        p_data_fp32.add_(-update)

    # if p.dtype in {torch.float16, torch.bfloat16}:
    #    p.copy_(p_data_fp32)

    if p.dtype == torch.bfloat16:
        if self.optimizer.use_kahan_summation:
            copy_kahan_(p, p_data_fp32, state, update)
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
