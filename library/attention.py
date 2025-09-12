import torch
from typing import Optional, Union

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    seq_lens: Optional[list[int]] = None,
    attn_mode: str = "torch",
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        seq_lens: Valid sequence length for each batch element.
        attn_mode: Attention implementation ("torch" or "sageattn").
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if seq_lens is None:
        seq_lens = [q.shape[1]] * q.shape[0]

    # Determine tensor layout based on attention implementation
    if attn_mode == "torch" or attn_mode == "sageattn":
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations

    # Process each batch element with its valid sequence length
    q_seq_len = q.shape[1]
    q = [transpose_fn(q[i : i + 1, : seq_lens[i]]) for i in range(len(q))]
    k = [transpose_fn(k[i : i + 1, : seq_lens[i]]) for i in range(len(k))]
    v = [transpose_fn(v[i : i + 1, : seq_lens[i]]) for i in range(len(v))]

    if attn_mode == "torch":
        x = []
        for i in range(len(q)):
            x_i = torch.nn.functional.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate)
            q[i] = None
            k[i] = None
            v[i] = None
            x.append(torch.nn.functional.pad(x_i, (0, 0, 0, q_seq_len - x_i.shape[2]), value=0))  # Pad to max seq len, B, H, L, D
        x = torch.cat(x, dim=0)
        del q, k, v

    elif attn_mode == "xformers":
        x = []
        for i in range(len(q)):
            x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)
            q[i] = None
            k[i] = None
            v[i] = None
            x.append(torch.nn.functional.pad(x_i, (0, 0, 0, 0, 0, q_seq_len - x_i.shape[1]), value=0))  # B, L, H, D
        x = torch.cat(x, dim=0)
        del q, k, v

    else:
        # Currently only PyTorch SDPA and xformers are implemented
        raise ValueError(f"Unsupported attention mode: {attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]
    return x
