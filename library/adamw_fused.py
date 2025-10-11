import math
import torch

from library.adafactor_fused import copy_stochastic_
from library.adafactor_fused import copy_kahan_


def to_float24_bytes(tensor_f32: torch.Tensor) -> torch.Tensor:
    """
    Converts a float32 tensor to a 'float24' representation for storage.

    This is done by taking the 3 most significant bytes of each float32 element.
    On a little-endian system, these are the last 3 bytes.
    # TODO - Check this works on Mac, which is a big-endian system

    Args:
        tensor_f32: The input tensor with dtype torch.float32.

    Returns:
        A 1D tensor of dtype torch.uint8 containing the packed 'float24' data.
    """
    if tensor_f32.dtype != torch.float32:
        raise TypeError("Input tensor must be of dtype torch.float32")

    tensor_u8 = tensor_f32.view(torch.uint8)
    tensor_u8_reshaped = tensor_u8.view(-1, 4)
    tensor_f24_bytes = tensor_u8_reshaped[:, 1:]
    return tensor_f24_bytes.flatten()


def from_float24_bytes(tensor_f24_u8: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Restores a 'float24' byte tensor back to a float32 tensor.

    Args:
        tensor_f24_u8: A 1D tensor of dtype torch.uint8 from to_float24_bytes.
        original_shape: The shape of the original float32 tensor.
        device: The device to create the restored tensor on.

    Returns:
        The restored tensor with dtype torch.float32 and the original shape.
    """
    if tensor_f24_u8.dtype != torch.uint8:
        raise TypeError("Input byte tensor must be of dtype torch.uint8")
    if tensor_f24_u8.numel() % 3 != 0:
        raise ValueError("Input byte tensor size must be a multiple of 3")
    
    tensor_u8_3bytes = tensor_f24_u8.view(-1, 3)
    padding = torch.zeros(tensor_u8_3bytes.shape[0], 1, dtype=torch.uint8, device=tensor_u8_3bytes.device)
    tensor_u8_4bytes = torch.cat([padding, tensor_u8_3bytes], dim=1)
    tensor_f32_flat = tensor_u8_4bytes.flatten().view(torch.float32)
    return tensor_f32_flat.view(original_shape)


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

        data_type = torch.float32 if high_quality else torch.uint16

        state['exp_avg']    = torch.zeros_like(p, dtype=data_type)
        state['exp_avg_sq'] = torch.zeros_like(p, dtype=data_type)

    state["step"] += 1

    # NAdam

    beta1, beta2 = group['betas']
    eps = group['eps']  # 1e-8
    weight_decay = group.get('weight_decay', 0.0)

    # Bias correction terms
    bias_correction1 = 1.0 - math.pow(beta1, state['step'])
    bias_correction2 = 1.0 - math.pow(beta2, state['step'])
        
    eps_p2: float = math.pow(eps, 2)

    # Bring state back (from CPU, if necessary)

    # Recover the exp avg states from however they're stored
    def unpack_tensor(state, key, target_device):

        # Stored as f24 format?
        if state[f'{key}'].dtype == torch.uint8:
            return from_float24_bytes(state[f'{key}'].to(target_device), state[f'{key}_shape'])

        # bf16 / u16 / f32
        return state[f'{key}'].to(target_device).to(dtype=torch.float32)

    state['exp_avg']    = unpack_tensor(state, 'exp_avg',    p.device)
    state['exp_avg_sq'] = unpack_tensor(state, 'exp_avg_sq', p.device)
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
    
    # Implement 'cautious optimizer' from https://arxiv.org/pdf/2411.16085
    # The scaling factor - dividing by m.mean() - does not seem to work with parameter
    # groups, but it also appears to be an optional step, so it has been removed.
    m = (update * grad >= 0).to(grad.dtype)
    update = update * m #/ (m.mean() + eps)

    # Apply learning rate
    update.mul_(lr)

    # Apply weight decay
    if weight_decay != 0:
        p_data_fp32.mul_(1 - lr * weight_decay)

    # Reduce the size of large exp_avg and exp_avg_sq tensors to 24-bit,
    # and then move them to cpu memory
    if not high_quality:
        state[f'exp_avg_shape'] = state[f'exp_avg'].shape
        state[f'exp_avg'] = to_float24_bytes(state[f'exp_avg']).to('cpu')

        state[f'exp_avg_sq_shape'] = state[f'exp_avg_sq'].shape
        state[f'exp_avg_sq'] = to_float24_bytes(state[f'exp_avg_sq']).to('cpu')

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
