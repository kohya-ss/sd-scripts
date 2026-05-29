import math
import torch

from library.adafactor_fused import copy_stochastic_
from library.adafactor_fused import copy_kahan_


# Pack floating point tensors into uint16. Their float32 bytes are interpreted as uint32
# bytes (not cast to uint32). Since positive floats are in sequential order when interpreted
# as uint32s, the groups of positive and negative floats appear as small ranges in uint32
# format. The three clumps (negative floats, zeros, postive floats) then have their min/max
# positions noted, and stretched to cover a uint16 range.
def pack_tensor(state, key, support_neg):

    k = state[f'{key}']
    k_uint32_f = torch.abs(k).view(torch.uint32).to(torch.float32)
    
    min_val, max_val = torch.aminmax(k_uint32_f[k_uint32_f != 0.0])

    # No support_neg (i.e. input floats are only zero or positive). Outputs values in these uint16 ranges:
    #            0 <-- 0.0s
    #     1..65535 <-- positive floats

    # support_neg (i.e. input floats can be zero or +/-). Outputs values in these uint16 ranges:
    #            0 <-- 0.0s
    #     1..32767 <-- positive floats
    #        32768 <-- -0.0 ?  Not used.
    # 32769..65535 <-- negative floats

    range = 32768 if support_neg else 65536

    k_int32_scale = (k_uint32_f - min_val) * (range - 2) / (max_val - min_val) + 1  # Scale into [1..range]

    packed = torch.where(k > 0, k_int32_scale, 0)  # Positive floats and zero
    if support_neg:
        packed = torch.where(k < 0, k_int32_scale + 32768, packed)  # Negative floats
    del k_int32_scale

    k_uint16_scale = packed.to(torch.uint16)

    state[f'{key}']     = k_uint16_scale
    state[f'{key}_min'] = min_val
    state[f'{key}_max'] = max_val

    pass


# Recover adan state tensors packed wtih pack_tensor()
def unpack_tensor(state, key, support_neg):

    # uint16 format = packed floats
    if state[f'{key}'].dtype == torch.uint16:
        packed  = state[f'{key}'].to('cuda').to(dtype=torch.float32)
        min_val = state[f'{key}_min']
        max_val = state[f'{key}_max']
        
        range = 32768.0 if support_neg else 65536.0

        if support_neg:
            pack_merge_signs = torch.where(packed >= 32768, packed - 32768, packed)
        else:
            pack_merge_signs = packed
        upck = (pack_merge_signs - 1) / (range - 2) * (max_val - min_val) + min_val
        upck = torch.where(pack_merge_signs == 0, 0, upck)  # 0's are special cased
        upck = upck.to(torch.uint32)
        upck_final_but_no_negs = upck.view(torch.float32)
        if support_neg:
            upck_final = torch.where(packed >= 32768, -upck_final_but_no_negs, upck_final_but_no_negs)
        else:
            upck_final = upck_final_but_no_negs

        return upck_final

    # bf16 / f32
    return state[f'{key}'].to('cuda').to(dtype=torch.float32)


@torch.no_grad()
def adan_offload_step_param(self, p, group):

    if p.grad is None:
        return
    grad = p.grad
    if grad.dtype in {torch.float16, torch.bfloat16}:
        grad = grad.float()
    if grad.is_sparse:
        raise RuntimeError("This Adan implementation does not support sparse gradients.")

    state = self.state[p]
    grad_shape = grad.shape

    p_data_fp32 = p
    if p.dtype in {torch.float16, torch.bfloat16}:
        p_data_fp32 = p_data_fp32.float()
    
    # Tensors with few elements may be more sensitive to quantization
    # errors, so keep them in float32
    #global tot_4096, tot_all
    high_quality = torch.numel(p) <= 2000000

    # State Initialization
    if len(state) == 0:
        state["step"] = 0

        state['exp_avg']      = torch.zeros_like(p, dtype=torch.float32 if high_quality else torch.bfloat16)
        state['exp_avg_sq']   = torch.zeros_like(p, dtype=torch.float32 if high_quality else torch.bfloat16)
        state['exp_avg_diff'] = torch.zeros_like(p, dtype=torch.float32 if high_quality else torch.bfloat16)
        state['neg_grad_or_diff'] = torch.zeros_like(p, dtype=torch.float32 if high_quality else torch.bfloat16)
    else:
        pass

    state["step"] += 1

    #beta1, beta2, beta3 = group['betas']  # Don't have custom class, so beta3 not available
    beta1, beta2, beta3 = (0.98, 0.92, 0.99)  # Hard coded betas for now
    eps = group['eps']  # 1e-8
    weight_decay = group.get('weight_decay', 0.0)  # Not currently implemented

    # Bias correction terms
    bias_correction1 = 1.0 - math.pow(beta1, state['step'])
    bias_correction2 = 1.0 - math.pow(beta2, state['step'])
    bias_correction3 = 1.0 - math.pow(beta3, state['step'])
    bias_correction3_sqrt = math.sqrt(bias_correction3)
        
    eps_p2: float = math.pow(eps, 2)

    # Recover the exp avg states from however they're stored
    state['exp_avg']      = unpack_tensor(state, 'exp_avg',      True)
    state['exp_avg_sq']   = unpack_tensor(state, 'exp_avg_sq',   False)
    state['exp_avg_diff'] = unpack_tensor(state, 'exp_avg_diff', True)
    state['neg_grad_or_diff'] = unpack_tensor(state, 'neg_grad_or_diff', True)

    exp_avg          = state['exp_avg']
    exp_avg_sq       = state['exp_avg_sq']
    exp_avg_diff     = state['exp_avg_diff']
    neg_grad_or_diff = state['neg_grad_or_diff']

    # for memory saving, we use `neg_grad_or_diff`
    # to get some temp variable in a inplace way
    neg_grad_or_diff.add_(grad)

    exp_avg     .mul_(beta1).add_(grad,             alpha= 1 - beta1)  # m_t
    exp_avg_diff.mul_(beta2).add_(neg_grad_or_diff, alpha= 1 - beta2)  # diff_t

    neg_grad_or_diff.mul_(beta2).add_(grad)
    exp_avg_sq      .mul_(beta3).addcmul_(neg_grad_or_diff, neg_grad_or_diff, value= 1 - beta3)  # n_t

    lr: float = group['lr']

    denom = (exp_avg_sq.sqrt() / bias_correction3_sqrt).add_(eps)
    step_size      = lr         / bias_correction1
    step_size_diff = lr * beta2 / bias_correction2

    # todo: weight decay not supported
    update  = (exp_avg      * step_size     ) / denom
    update += (exp_avg_diff * step_size_diff) / denom

    neg_grad_or_diff.zero_().add_(grad, alpha=-1.0)

    # Just build momentum for first few steps
    if state['step'] <= 3:
        update.mul_(0.0)

    # Move the optimizer state tensors to main memory
    if not high_quality:

        # float32 to uint16 compression, hopefully provides more precision
        pack_tensor(state, 'exp_avg',      True)
        pack_tensor(state, 'exp_avg_sq',   False)  # Only positive floats
        pack_tensor(state, 'exp_avg_diff', True)

        state[f'exp_avg']      = state[f'exp_avg']     .to('cpu')
        state[f'exp_avg_sq']   = state[f'exp_avg_sq']  .to('cpu')
        state[f'exp_avg_diff'] = state[f'exp_avg_diff'].to('cpu')

        # Neg_grad is always a bfloat16 (stored in a float32) already apparently! So
        # can be stored as a bfloat16 exactly.
        state[f'neg_grad_or_diff'] = state[f'neg_grad_or_diff'].to(torch.bfloat16).to('cpu')

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
def adan_offload_step(self, closure=None):
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
            adan_offload_step_param(self, p, group)

    return loss


def patch_adan_offload_fused(optimizer, use_nesterov):
    optimizer.use_nesterov = use_nesterov

    optimizer.step_param = adan_offload_step_param.__get__(optimizer)
    optimizer.step = adan_offload_step.__get__(optimizer)
