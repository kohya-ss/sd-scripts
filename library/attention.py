# Unified attention function supporting various implementations

from dataclasses import dataclass
import torch
from typing import List, Optional, Union

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[List[int]] = None  # valid length (img_len + text length) of each batch element
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None
    # Set when all batch elements have the same valid length and split_attn is False: `attention` trims the
    # sequence to this length instead of masking. Python ints are used (not tensors) so that no graph break
    # occurs inside torch.compile'd blocks; torch.compile treats them as dynamic after the first recompile.
    uniform_seqlen: Optional[int] = None
    # Key / value side when the keys have extra (always valid) tokens inserted after the image tokens (`kv_img_len` >
    # `img_len`, e.g. cached reference K/V): valid length, cumulative lengths, padded length and uniform length of the
    # keys. None means the same as the query side.
    kv_seqlens: Optional[List[int]] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_k: Optional[int] = None
    uniform_kv_seqlen: Optional[int] = None

    @property
    def supports_fp32(self) -> bool:
        return self.attn_mode not in ["flash"]

    @property
    def requires_same_dtype(self) -> bool:
        return self.attn_mode in ["xformers"]

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str],
        split_attn: bool,
        img_len: Optional[int],
        attention_mask: Optional[torch.Tensor],
        kv_img_len: Optional[int] = None,
    ) -> "AttentionParams":
        """
        Args:
            img_len: number of (always valid) image tokens before the right-padded text tokens of the queries
            attention_mask: [B, L] mask of the text tokens (1 = valid), None if all tokens are valid
            kv_img_len: number of always valid tokens before the text tokens on the key / value side, if the keys have
                extra tokens inserted after the image tokens (e.g. cached reference K/V). None: same as img_len.
        """
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid (the keys may still be longer than the queries)
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            # The sequence lengths are converted to Python ints here, once per model forward (one device sync),
            # so that the per-block `attention` calls do not need any data-dependent control flow.
            seqlens = [int(n) + img_len for n in attention_mask.sum(dim=1).tolist()]  # [B]
            max_seqlen = attention_mask.shape[1] + img_len

            kv_extra = 0 if kv_img_len is None else kv_img_len - img_len
            assert kv_extra >= 0, "kv_img_len must not be smaller than img_len"
            kv_seqlens = [n + kv_extra for n in seqlens] if kv_extra > 0 else None
            max_seqlen_k = max_seqlen + kv_extra if kv_extra > 0 else None

            if split_attn:
                # cu_seqlens is not needed for split attention
                return AttentionParams(
                    attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen, None, kv_seqlens, None, max_seqlen_k
                )

            # All batch elements have the same valid length: `attention` trims the sequence instead of masking
            uniform_seqlen = seqlens[0] if all(n == seqlens[0] for n in seqlens) else None
            uniform_kv_seqlen = uniform_seqlen + kv_extra if (uniform_seqlen is not None and kv_extra > 0) else None

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens_list = [0]
            cu_seqlens_k_list = [0]
            for i in range(batch_size):
                cu_seqlens_list.append(i * max_seqlen + seqlens[i])  # end of valid tokens for query
                cu_seqlens_list.append((i + 1) * max_seqlen)  # end of all tokens for query
                if kv_extra > 0:
                    cu_seqlens_k_list.append(i * max_seqlen_k + kv_seqlens[i])
                    cu_seqlens_k_list.append((i + 1) * max_seqlen_k)
            cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=attention_mask.device)
            cu_seqlens_k = (
                torch.tensor(cu_seqlens_k_list, dtype=torch.int32, device=attention_mask.device) if kv_extra > 0 else None
            )

            # Expand attention mask to include the image (and extra key) tokens: the mask is applied to the keys
            attention_mask = torch.nn.functional.pad(attention_mask, (img_len + kv_extra, 0), value=1)  # [B, kv_img_len + L]

            if attn_mode == "xformers":
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens, kv_seqlens if kv_seqlens is not None else seqlens, device=attention_mask.device
                )
            elif attn_mode == "torch":
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, kv_img_len + L]

            return AttentionParams(
                attn_mode,
                split_attn,
                img_len,
                attention_mask,
                seqlens,
                cu_seqlens,
                max_seqlen,
                uniform_seqlen,
                kv_seqlens,
                cu_seqlens_k,
                max_seqlen_k,
                uniform_kv_seqlen,
            )


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, Lk, H, D]. Lk may be larger than L (extra key tokens, see `kv_img_len` of AttentionParams).
        v: Value tensor [B, Lk, H, D].
        attn_params: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # If split attn is False, attention mask is provided and all sequence lengths are same, we can trim the sequence.
    # uniform_seqlen is a Python int computed in create_attention_params_from_mask (no tensor comparison here, so
    # this does not cause a graph break under torch.compile)
    seqlen_trimmed = False
    if not attn_params.split_attn and attn_params.attention_mask is not None and attn_params.uniform_seqlen is not None:
        seqlen = attn_params.uniform_seqlen
        kv_seqlen = attn_params.uniform_kv_seqlen if attn_params.uniform_kv_seqlen is not None else seqlen
        if seqlen < q.shape[1]:
            q = q[:, :seqlen]
            seqlen_trimmed = True
        if kv_seqlen < k.shape[1]:
            k = k[:, :kv_seqlen]
            v = v[:, :kv_seqlen]
        max_seqlen = attn_params.max_seqlen
        attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)  # do not in-place modify
        attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA and sageattn with fixed length
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0)

    # Process each batch element with its valid sequence lengths.
    # Note: the list comprehensions below capture q/k/v, which makes them closure cells on Python < 3.12 (PEP 709
    # inlines comprehensions since 3.12). `del q, k, v` would then compile to DELETE_DEREF, which dynamo does not
    # support (graph break in every compiled block), so the references are released with `q = k = v = None` instead.
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # If no seqlens provided, assume all tokens are valid
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, True)  # do not in-place modify
            attn_params.seqlens = [q.shape[1]] * q.shape[0]
            attn_params.max_seqlen = q.shape[1]
            if k.shape[1] != q.shape[1]:
                attn_params.kv_seqlens = [k.shape[1]] * k.shape[0]
        kv_seqlens = attn_params.kv_seqlens if attn_params.kv_seqlens is not None else attn_params.seqlens
        q = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))]
        k = [transpose_fn(k[i : i + 1, : kv_seqlens[i]]) for i in range(len(k))]
        v = [transpose_fn(v[i : i + 1, : kv_seqlens[i]]) for i in range(len(v))]
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    if attn_params.attn_mode == "torch":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = torch.nn.functional.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q = k = v = None

        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate)
            q = k = v = None

    elif attn_params.attn_mode == "xformers":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q = k = v = None

        else:
            x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate)
            q = k = v = None

    elif attn_params.attn_mode == "sageattn":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(q[i], k[i], v[i])  # B, H, L, D. No dropout support
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q = k = v = None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v)  # B, L, H, D. No dropout support
            q = k = v = None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # No dropout support
            cu_seqlens_k = attn_params.cu_seqlens_k if attn_params.cu_seqlens_k is not None else attn_params.cu_seqlens
            max_seqlen_k = attn_params.max_seqlen_k if attn_params.max_seqlen_k is not None else attn_params.max_seqlen
            x = sageattn_varlen(q, k, v, attn_params.cu_seqlens, cu_seqlens_k, attn_params.max_seqlen, max_seqlen_k)
            q = k = v = None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = flash_attn_func(q[i], k[i], v[i], drop_rate)  # B, L, H, D
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q = k = v = None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = flash_attn_func(q, k, v, drop_rate)  # B, L, H, D
            q = k = v = None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            cu_seqlens_k = attn_params.cu_seqlens_k if attn_params.cu_seqlens_k is not None else attn_params.cu_seqlens
            max_seqlen_k = attn_params.max_seqlen_k if attn_params.max_seqlen_k is not None else attn_params.max_seqlen
            x = flash_attn_varlen_func(
                q, k, v, attn_params.cu_seqlens, cu_seqlens_k, attn_params.max_seqlen, max_seqlen_k, drop_rate
            )
            q = k = v = None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    else:
        # Currently only PyTorch SDPA and xformers are implemented
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen

    return x
