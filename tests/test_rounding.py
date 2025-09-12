import torch
from library.rounding_util import (
    round_tensor_det,
    round_tensor_stoch,
    round_parameters,
    fake_quantize,
    compute_per_channel_step,
    fake_quantize_levels,
    compute_scale_bits,
)


def test_round_tensor_det_basic():
    x = torch.tensor([0.05, 0.14, -0.26, 1.24, -1.25], dtype=torch.float32)
    y = round_tensor_det(x, 0.1)
    expected = torch.tensor([0.1, 0.1, -0.3, 1.2, -1.2], dtype=torch.float32)
    assert torch.allclose(y, expected)


def test_round_tensor_stoch_range_and_grid():
    torch.manual_seed(123)
    step = 0.5
    x = torch.tensor([0.2, 0.4, 0.6, -0.2, -0.6], dtype=torch.float32)
    y = round_tensor_stoch(x, step)
    # Values must lie on step grid
    assert torch.all(((y / step) - torch.round(y / step)).abs() < 1e-6)
    # And be within one step of original
    assert torch.all((y - x).abs() <= step)


def test_round_parameters_inplace_and_dtype():
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.tensor([0.03, 0.27], dtype=torch.float16))
    p2 = torch.nn.Parameter(torch.tensor([-0.21, 1.01], dtype=torch.bfloat16))
    round_parameters([p1, p2], step=0.1, mode="det")
    # Check dtype preserved and values on grid
    assert p1.data.dtype == torch.float16
    assert p2.data.dtype == torch.bfloat16
    for p in (p1, p2):
        grid = (p.data.to(torch.float32) / 0.1) - torch.round(p.data.to(torch.float32) / 0.1)
        assert torch.all(grid.abs() < 1e-3)


def test_fake_quantize_ste_gradient_det():
    x = torch.tensor([0.05, 0.14, -0.26], dtype=torch.float32, requires_grad=True)
    y = fake_quantize(x, step=0.1, mode="det")
    y.sum().backward()
    # STE should pass gradient ~1
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-6)


def test_per_channel_step_rms_shape_and_broadcast():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 4)
    base = 1e-2
    step = compute_per_channel_step(x, base, stat="rms")
    # shape broadcastable to x
    assert step.shape == (1, 3, 1, 1)
    y = fake_quantize(x, step=step, mode="det")
    # grid check per channel
    z = (y / step) - torch.round(y / step)
    assert torch.all(z.abs() < 1e-4)


def test_fake_quantize_levels_sym8bit_basic():
    x = torch.tensor([-1.23, -0.11, 0.0, 0.12, 1.01], dtype=torch.float32, requires_grad=True)
    bits = 8
    qmax = (1 << (bits - 1)) - 1
    # per-tensor scale using absmax
    scale = compute_scale_bits(x, bits=bits, granularity="tensor", stat="absmax")
    y = fake_quantize_levels(x, scale=scale, qmin=-qmax, qmax=qmax, mode="det")
    # check that y/scale lies on integer grid and is clamped
    g = (y / scale) - torch.round(y / scale)
    assert torch.all(g.abs() < 1e-5)
    assert torch.all((y / scale) <= qmax + 1e-5)
    assert torch.all((y / scale) >= -qmax - 1e-5)
    # STE gradient passes ~1
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-6)
