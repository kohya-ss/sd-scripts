import math
import torch

from library.adafactor_fused import copy_stochastic_
from library.adafactor_fused import copy_kahan_


@torch.no_grad()
def adamw_offload_step_param(self, p, group):

    if p.grad is None:
        return
    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("This (N)AdamW implementation does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()
    
    # Tensors with few elements may be more sensitive to quantization
    # errors, so keep them in float32
    high_quality = torch.numel(p) <= 4096

    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        if high_quality:
            # Exponential averages stored in f32 format
            state['exp_avg']    = torch.zeros_like(p, dtype=torch.float32)
            state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
        else:
            # Exponential averages stored in u16 format
            state['exp_avg'] = torch.zeros_like(p, dtype=torch.uint16)
            state['exp_avg_min'] = 0.0
            state['exp_avg_max'] = 1.0

            state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.uint16)
            state['exp_avg_sq_min'] = 0.0
            state['exp_avg_sq_max'] = 1.0

    state["step"] += 1

    # NAdam

    beta1, beta2 = group['betas']
    eps = group['eps']  # 1e-8
    weight_decay = group.get('weight_decay', 0.0)

    # Bias correction terms
    bias_correction1 = 1.0 - math.pow(beta1, state['step'])
    bias_correction2 = 1.0 - math.pow(beta2, state['step'])
        
    eps_p2: float = math.pow(eps, 2)

    # Bring state back from CPU

    if high_quality:
        # These exponential averages are already in float32 format
        state['exp_avg']    = state['exp_avg']   .to(p.device)
        state['exp_avg_sq'] = state['exp_avg_sq'].to(p.device)
    else:
        # Unpack these exponential averages from uint16 format

        # A power function was applied to the tensor values, as they are usually
        # distributed in an exponential fashion. After the power function was applied,
        # the min and max of the results were noted, and then the values were scaled
        # to the 0-65535 range for storage. This process is reversed here.

        u16power = 8.0  # This value worked acceptably in testing to spread the values more evenly

        exp_avg_min = state['exp_avg_min']
        exp_avg_max = state['exp_avg_max']
        exp_avg_sq_min = state['exp_avg_sq_min']
        exp_avg_sq_max = state['exp_avg_sq_max']

        uint16_recreate_a = state['exp_avg'].to(p.device).to(dtype=torch.float32) / 65535.0 * (exp_avg_max - exp_avg_min) + exp_avg_min
        state['exp_avg'] = torch.pow(torch.abs(uint16_recreate_a), u16power) * torch.sgn(uint16_recreate_a)
        del uint16_recreate_a

        uint16_recreate_a_sq = state['exp_avg_sq'].to(p.device).to(dtype=torch.float32) / 65535.0 * (exp_avg_sq_max - exp_avg_sq_min) + exp_avg_sq_min
        state['exp_avg_sq'] = torch.pow(torch.abs(uint16_recreate_a_sq), u16power) * torch.sgn(uint16_recreate_a_sq)
        del uint16_recreate_a_sq

    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

    # Update biased first and second moment estimates
    exp_avg   .mul_(beta1).add_    (grad,       alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # Compute bias-corrected second moment for denominator
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2

    # Compute update based on whether Nesterov momentum (NAdam) is being used
    if self.use_nesterov:
        # The next step's bias correction for momentum is needed
        bias_correction1_next = 1.0 - math.pow(beta1, state['step'] + 1)

        # NAdam update: combines current gradient with momentum look-ahead
        momentum_cache = exp_avg / bias_correction1_next
        update = (beta1 * momentum_cache + (1.0 - beta1) * grad / bias_correction1) / (exp_avg_sq_corrected.sqrt() + eps)
    else:
        # Standard Adam update: use bias-corrected first moment directly
        exp_avg_corrected = exp_avg / bias_correction1
        update = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)

    lr: float = group['lr']
    
    # Apply learning rate
    update.mul_(lr)

    # Apply weight decay
    if weight_decay != 0:
        p_data_fp32.mul_(1 - lr * weight_decay)

    if high_quality:

        # These are kept in f32 format between steps
        state['exp_avg'] = state['exp_avg'].to('cpu')
        state['exp_avg_sq'] = state['exp_avg_sq'].to('cpu')

    else:

        # Compress the exp_avg and exp_avg_sq tensors to cut their size down
        # from 32 bit to 16 bit.
        #
        # A power function is applied to try to linearize the tensor values, as
        # they are usually distributed in an exponential fashion. It would have
        # been preferable to use a log() function, but the input values can be
        # negative, so a pow() function is used instead. The 1/16th power was
        # chosen fairly arbitrarily, but seemed to distribute the values fairly
        # reasonably in some simple tests.
        # 
        # After the power function is applied, the min and max of the resulting
        # values are stored, and the values are then scaled to the 0-65535 range
        # for storage.
        #
        # Doing this instead of storing these values as bf16 reduced the L1
        # error between the stored values and the true f32 values by around 90%,
        # with a notable increase in output image quality.

        log_exp_avg = torch.pow(torch.abs(state['exp_avg']), 1.0 / u16power) * torch.sgn(state['exp_avg'])
        exp_avg_min = torch.min(log_exp_avg)
        exp_avg_max = torch.max(log_exp_avg)
        state['exp_avg_min'] = exp_avg_min
        state['exp_avg_max'] = exp_avg_max
        normalized = (log_exp_avg - exp_avg_min) / (exp_avg_max - exp_avg_min)
        del log_exp_avg

        state['exp_avg'] = (normalized * 65535.0).clamp(0, 65535).to(dtype=torch.uint16).to('cpu')

        log_exp_avg_sq = torch.pow(torch.abs(state['exp_avg_sq']), 1.0 / u16power) * torch.sgn(state['exp_avg_sq'])
        exp_avg_sq_min = torch.min(log_exp_avg_sq)
        exp_avg_sq_max = torch.max(log_exp_avg_sq)
        state['exp_avg_sq_min'] = exp_avg_sq_min
        state['exp_avg_sq_max'] = exp_avg_sq_max
        normalized_sq = (log_exp_avg_sq - exp_avg_sq_min) / (exp_avg_sq_max - exp_avg_sq_min)
        del log_exp_avg_sq

        state['exp_avg_sq'] = (normalized_sq * 65535.0).clamp(0, 65535).to(dtype=torch.uint16).to('cpu')

    # Add on gradient update, but not if using kahan summation as the bottom
    # bits must be restored first. (This update occurs in copy_kahan_() instead)
    if not self.optimizer.use_kahan_summation:
        p_data_fp32.add_(-update)

    if p.dtype == torch.bfloat16:
        if self.optimizer.use_kahan_summation:
            copy_kahan_(p, p_data_fp32, state, update)
        else:
            copy_stochastic_(p, p_data_fp32)
    elif p.dtype == torch.float16:
        p.copy_(p_data_fp32)


@torch.no_grad()
def adamw_offload_step(self, closure=None):
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
            adamw_offload_step_param(self, p, group)

    return loss


def patch_adamw_offload_fused(optimizer, use_nesterov):
    optimizer.use_nesterov = use_nesterov

    optimizer.step_param = adamw_offload_step_param.__get__(optimizer)
    optimizer.step = adamw_offload_step.__get__(optimizer)
