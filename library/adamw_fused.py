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
    
    # State Initialization
    if len(state) == 0:
        state["step"] = 0
        state['exp_avg'] = torch.zeros_like(p, dtype=torch.bfloat16)
        state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.bfloat16)

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
    state['exp_avg']    = state['exp_avg']   .to('cuda').to(dtype=torch.float32)
    state['exp_avg_sq'] = state['exp_avg_sq'].to('cuda').to(dtype=torch.float32)
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

    # Keep state on CPU
    state['exp_avg']    = state['exp_avg']   .to(dtype=torch.bfloat16).to('cpu')
    state['exp_avg_sq'] = state['exp_avg_sq'].to(dtype=torch.bfloat16).to('cpu')

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
