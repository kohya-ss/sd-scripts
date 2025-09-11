import torch
from typing import Optional


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: list[int], attn_mode: str = "torch", drop_rate: float = 0.0
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        q: Query tensor [B, L, H, D].
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        seq_lens: Valid sequence length for each batch element.
        attn_mode: Attention implementation ("torch" or "sageattn").
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    # Determine tensor layout based on attention implementation
    if attn_mode == "torch" or attn_mode == "sageattn":
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations

    # Process each batch element with its valid sequence length
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
            x.append(x_i)
        x = torch.cat(x, dim=0)
        del q, k, v
    # Currently only PyTorch SDPA is implemented

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]
    return x
