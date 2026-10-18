# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from comfy-kitchen (https://github.com/Comfy-Org/comfy-kitchen):
# - backends/triton/quantization.py (INT8 section, itself derived from dxqb/OneTrainer
#   and ComfyUI-Flux2-INT8)
# - tensor/int8_utils.py (Hadamard rotation helpers)
# - backends/eager/quantization.py (offline row-wise INT8 quantization)
# Modifications (Musubi Tuner, ported to sd-scripts as is): triton made an optional import
# (eager fallback when unavailable), stochastic rounding removed, unused formats
# (fp8/nvfp4/mxfp8) dropped, helpers inlined to make the file self-contained.

"""ConvRot INT8 kernels: block-diagonal Hadamard rotation + row-wise INT8 quantization.

ConvRot (arXiv:2512.03673) smooths activation outliers by rotating each group of
`group_size` input features with a normalized regular Hadamard matrix H (symmetric,
orthogonal: H @ H = I) before quantization:

- offline:  W_rot = W @ H^T (per group), then per-output-channel INT8 quantization
- online:   x_rot = x @ H (per group), dynamic per-row INT8 quantization, INT8 GEMM
            with the dequantization epilogue fused into the matmul kernel

Since H is symmetric orthogonal, y = x_rot @ W_rot^T equals x @ W^T up to
quantization error.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    # The fused kernels are CUDA only, and `triton.autotune` needs an active GPU driver when the kernels below are
    # defined (import time): on a CPU-only machine with triton installed (e.g. CI) it raises RuntimeError
    # ("0 active drivers"), not ImportError. Fall back to the eager implementation in that case.
    try:
        from triton.runtime import driver as _triton_driver

        HAS_TRITON = torch.cuda.is_available() and _triton_driver.active is not None
    except Exception:
        HAS_TRITON = False


# =============================================================================
# Hadamard rotation helpers (from comfy_kitchen/tensor/int8_utils.py)
# =============================================================================

_HADAMARD_CACHE = {}


def _build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot)."""
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )

    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4

    h_normalized = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h_normalized
    return h_normalized


def _rotate_weight(
    weight: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T."""
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    weight_grouped = weight.reshape(out_f, n_groups, group_size)
    h_t = h.T.to(dtype=weight.dtype, device=weight.device)
    weight_rotated = torch.matmul(weight_grouped, h_t)
    return weight_rotated.reshape(out_f, in_f)


def _rotate_activation(
    x: torch.Tensor,
    h: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate activation online using Optimized Matmul implementation."""
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by group_size {group_size}")
    n_groups = features // group_size

    x_grouped = x.reshape(-1, n_groups, group_size)
    h = h.to(dtype=x.dtype, device=x.device)
    x_rotated = torch.matmul(x_grouped, h)

    return x_rotated.reshape(orig_shape)


# =============================================================================
# Offline row-wise INT8 quantization (from comfy_kitchen/backends/eager/quantization.py,
# stochastic rounding removed)
# =============================================================================


def _int8_scale_for_math(scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    scale = scale.to(device=x.device, dtype=x.dtype)
    scale_min = torch.finfo(x.dtype).tiny
    return torch.where(scale == 0, torch.full_like(scale, scale_min), scale)


def _round_int8(scaled: torch.Tensor) -> torch.Tensor:
    return scaled.round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_int8_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales.

    Returns:
        Tuple of (quantized_int8 [..., K], scales float32 [..., 1]).
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = _round_int8(x / _int8_scale_for_math(scale, x))
    return q, scale


def quantize_int8_convrot_weight(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight rotation followed by row-wise INT8 quantization.

    The rotation is computed in float32 regardless of the weight dtype: rotating in
    bf16 loses enough precision to shift ~9% of the int8 codes by ±1-2, while fp32
    rotation reproduces ComfyUI's published pre-quantized checkpoints bit-exactly
    (verified against the MiniMax-H3 INT8 ConvRot distribution).

    Returns:
        Tuple of (rotated quantized weight int8 [N, K], scale float32 [N, 1]).
    """
    h = _build_hadamard(group_size, device=weight.device, dtype=torch.float32)
    weight_rot = _rotate_weight(weight.to(torch.float32), h, group_size)
    return quantize_int8_rowwise(weight_rot)


def dequantize_int8_convrot_weight(q: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dequantize INT8 ConvRot weights and rotate them back to the original basis."""
    h = _build_hadamard(group_size, device=q.device, dtype=torch.float32)
    return _rotate_weight(q.float() * scale, h, group_size)


# =============================================================================
# Triton kernels (from comfy_kitchen/backends/triton/quantization.py INT8 section:
# per-row activation quantization + INT8 GEMM with fused dequantization epilogue)
# =============================================================================

if HAS_TRITON:

    @triton.jit
    def _quantize_rowwise_kernel(
        x_ptr,  # Input pointer (FP16/BF16)
        y_ptr,  # Output pointer (INT8)
        s_ptr,  # Scale pointer (FP32)
        n_elements,  # Number of columns
        block_size: tl.constexpr,
        input_dtype_code: tl.constexpr,
    ):
        # Row index we are processing. Promote to int64 before the stride multiply:
        # H3 packs video+audio+text into one ~90k-row sequence, so row*cols can
        # exceed int32 (Triton wraps silently, corrupting the tail of the tensor).
        row_idx = tl.program_id(0).to(tl.int64)

        # Pointers to the start of the row
        x_row_ptr = x_ptr + row_idx * n_elements
        y_row_ptr = y_ptr + row_idx * n_elements

        # 1. Compute Max Abs Value for the row
        offsets = tl.arange(0, block_size)
        mask = offsets < n_elements

        # Load data
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)

        # Absolute value
        abs_x = tl.abs(x)

        # Reduction to find max
        max_val = tl.max(abs_x, axis=0)

        # 2. Compute Scale
        scale = tl.maximum(max_val / 127.0, 1e-30)

        # 3. Quantize
        if input_dtype_code == 1:
            q_f = (x / scale.to(tl.float16)).to(tl.float16)
        elif input_dtype_code == 2:
            q_f = (x / scale.to(tl.bfloat16)).to(tl.bfloat16)
        else:
            q_f = x / scale

        # Round and Clamp
        q_i = tl.clamp(libdevice.rint(q_f.to(tl.float32)), -128.0, 127.0).to(tl.int32)

        # 4. Store
        tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
        tl.store(s_ptr + row_idx, scale.to(tl.float32))

    def triton_quantize_rowwise(x: torch.Tensor):
        """
        Input: [Batch, Dim] (float16/bfloat16/float32)
        Output: [Batch, Dim] (int8), [Batch, 1] (float32)
        """
        rows, cols = x.shape
        y = torch.empty_like(x, dtype=torch.int8)
        s = torch.empty((rows, 1), device=x.device, dtype=torch.float32)

        input_dtype_code = 1 if x.dtype == torch.float16 else 2 if x.dtype == torch.bfloat16 else 0

        # Heuristic for block size
        block_size = triton.next_power_of_2(cols)
        if block_size < 128:
            block_size = 128

        grid = (rows,)
        _quantize_rowwise_kernel[grid](
            x,
            y,
            s,
            cols,
            block_size=block_size,
            input_dtype_code=input_dtype_code,
        )
        return y, s

    @triton.autotune(
        configs=[
            triton.Config({"block_m": 128, "block_n": 256, "block_k": 64, "group_size_m": 8}, num_stages=3, num_warps=8),
            triton.Config({"block_m": 64, "block_n": 256, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 64, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 64, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 32, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
        ],
        key=["m", "n", "k"],
    )
    @triton.jit
    def _int8_matmul_dequant_kernel(
        # Pointers
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        bias_ptr,
        # Matrix Dimensions
        m,
        n,
        k,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        group_size_m: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        """
        Computes: C = ((A * B) * (scale_a * scale_b)) + bias
        A: [m, k] int8
        B: [n, k] int8 (Transposed physically or logically via strides)
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(m, block_m)
        num_pid_n = tl.cdiv(n, block_n)
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + (pid % actual_group_size_m)
        pid_n = (pid % num_pid_in_group) // actual_group_size_m

        # 1. Prepare Pointers for A and B
        offs_am = (pid_m * block_m + tl.arange(0, block_m)) % m
        offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % n
        offs_k = tl.arange(0, block_k)

        # row * stride must be int64: at H3 scale m*stride_am and m*stride_cm exceed
        # int32 (e.g. fc1 output m~90k x n=28672), and Triton wraps i32 silently
        offs_am_i64 = offs_am.to(tl.int64)
        a_ptrs = a_ptr + (offs_am_i64[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # 2. Main Loop (Accumulate in Int32)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

        for k_idx in range(0, tl.cdiv(k, block_k)):
            # Load chunks
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k - k_idx * block_k, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k - k_idx * block_k, other=0)

            # Matrix Multiply
            accumulator += tl.dot(a, b)

            # Advance pointers
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk

        # 3. Fused Epilogue (Dequantize & Bias)
        scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]
        scale_b = tl.load(b_scale_ptr)

        c = accumulator.to(tl.float32)
        total_scale = scale_a[:, None] * scale_b
        c = c * total_scale

        if has_bias:
            bias = tl.load(bias_ptr + offs_bn)  # Vector [BLOCK_N]
            c = c + bias[None, :]

        # 4. Store Result
        c_ptrs = c_ptr + stride_cm * offs_am_i64[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < m) & (offs_bn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.autotune(
        configs=[
            triton.Config({"block_m": 128, "block_n": 256, "block_k": 64, "group_size_m": 8}, num_stages=3, num_warps=8),
            triton.Config({"block_m": 64, "block_n": 256, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 64, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 64, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 32, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
        ],
        key=["m", "n", "k"],
    )
    @triton.jit
    def _int8_matmul_dequant_per_row_kernel(
        # Pointers
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        bias_ptr,
        # Matrix Dimensions
        m,
        n,
        k,
        # Strides
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        # Meta-parameters
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        group_size_m: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        """
        Computes: C = ((A * B) * (scale_a[:, None] * scale_b[None, :])) + bias
        A: [m, k] int8, scale_a: [m, 1] per-row activation scales
        B: [n, k] int8, scale_b: [n, 1] per-row weight scales
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(m, block_m)
        num_pid_n = tl.cdiv(n, block_n)
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + (pid % actual_group_size_m)
        pid_n = (pid % num_pid_in_group) // actual_group_size_m

        # 1. Prepare Pointers for A and B
        offs_am = (pid_m * block_m + tl.arange(0, block_m)) % m
        offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % n
        offs_k = tl.arange(0, block_k)

        # row * stride must be int64: at H3 scale m*stride_am and m*stride_cm exceed
        # int32 (e.g. fc1 output m~90k x n=28672), and Triton wraps i32 silently
        offs_am_i64 = offs_am.to(tl.int64)
        a_ptrs = a_ptr + (offs_am_i64[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # 2. Main Loop (Accumulate in Int32)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)

        for k_idx in range(0, tl.cdiv(k, block_k)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k - k_idx * block_k, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k - k_idx * block_k, other=0)
            accumulator += tl.dot(a, b)
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk

        # 3. Fused Epilogue (Dequantize & Bias)
        scale_a = tl.load(a_scale_ptr + offs_am)  # Vector [BLOCK_M]
        scale_b = tl.load(b_scale_ptr + offs_bn)  # Vector [BLOCK_N]

        c = accumulator.to(tl.float32)
        total_scale = scale_a[:, None] * scale_b[None, :]
        c = c * total_scale

        if has_bias:
            bias = tl.load(bias_ptr + offs_bn)
            c = c + bias[None, :]

        # 4. Store Result
        c_ptrs = c_ptr + stride_cm * offs_am_i64[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < m) & (offs_bn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)

    def int8_linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype = torch.bfloat16,
        convrot: bool = False,
        convrot_groupsize: int = 256,
    ) -> torch.Tensor:
        """INT8 linear layer using fused Triton kernel.

        Quantizes x dynamically per-row, uses tensorwise or per-channel weight scale.

        Args:
            x: Input tensor [..., K].
            weight: INT8 weight tensor [N, K].
            weight_scale: Weight scale.
            bias: Optional bias [N].
            out_dtype: Output dtype.
            convrot: If True, apply online activation rotation.
            convrot_groupsize: Group size for Hadamard rotation.

        Returns:
            Result tensor [..., N].
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        m, k = x_2d.shape
        n = weight.shape[0]

        # Quantize input per-row using fused or normal quantization kernel
        if convrot:
            h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
            x_rotated = _rotate_activation(x_2d, h, convrot_groupsize)
            x_int8, x_scale = triton_quantize_rowwise(x_rotated)
        else:
            x_int8, x_scale = triton_quantize_rowwise(x_2d)

        output = torch.empty((m, n), device=x.device, dtype=out_dtype)

        is_per_channel = False
        if not isinstance(weight_scale, torch.Tensor):
            weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
        elif weight_scale.numel() == 1:
            weight_scale = weight_scale.reshape(1)
        else:
            is_per_channel = True
            weight_scale = weight_scale.reshape(n).contiguous()

        has_bias = bias is not None
        bias_ptr = bias if has_bias else x

        def grid(meta):
            return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)

        if is_per_channel:
            _int8_matmul_dequant_per_row_kernel[grid](
                a_ptr=x_int8,
                b_ptr=weight,
                c_ptr=output,
                a_scale_ptr=x_scale,
                b_scale_ptr=weight_scale,
                bias_ptr=bias_ptr,
                m=m,
                n=n,
                k=k,
                stride_am=x_int8.stride(0),
                stride_ak=x_int8.stride(1),
                stride_bk=weight.stride(1),
                stride_bn=weight.stride(0),
                stride_cm=output.stride(0),
                stride_cn=output.stride(1),
                has_bias=has_bias,
            )
        else:
            _int8_matmul_dequant_kernel[grid](
                a_ptr=x_int8,
                b_ptr=weight,
                c_ptr=output,
                a_scale_ptr=x_scale,
                b_scale_ptr=weight_scale,
                bias_ptr=bias_ptr,
                m=m,
                n=n,
                k=k,
                stride_am=x_int8.stride(0),
                stride_ak=x_int8.stride(1),
                stride_bk=weight.stride(1),
                stride_bn=weight.stride(0),
                stride_cm=output.stride(0),
                stride_cn=output.stride(1),
                has_bias=has_bias,
            )

        return output.reshape(*orig_shape[:-1], n)

else:
    triton_quantize_rowwise = None
    int8_linear = None
