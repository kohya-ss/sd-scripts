import builtins
from types import SimpleNamespace

import pytest
import torch

import networks.lora as lora_module
from networks.lora import (
    LoRANetwork,
    _compute_lora_effective_rank_stats,
    _compute_lora_effective_rank_stats_batched,
)
from train_network import _write_csv_rows


def _rank_lora(name, down, up, *, scale=1.0, multiplier=1.0):
    return SimpleNamespace(
        lora_name=name,
        lora_down=SimpleNamespace(weight=torch.nn.Parameter(down.clone())),
        lora_up=SimpleNamespace(weight=torch.nn.Parameter(up.clone())),
        scale=scale,
        multiplier=multiplier,
    )


def _assert_stats_close(actual, expected):
    assert actual["module"] == expected["module"]
    assert actual["r"] == expected["r"]
    assert isinstance(actual["r"], int)
    for key in ("sat", "top1", "energy"):
        assert isinstance(actual[key], float)
        assert actual[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-6)


def test_batched_rank_stats_match_scalar_for_mixed_shapes_ranks_and_conv():
    generator = torch.Generator().manual_seed(1234)
    loras = [
        _rank_lora(
            "linear_a",
            torch.randn(2, 3, generator=generator),
            torch.randn(4, 2, generator=generator),
        ),
        _rank_lora(
            "linear_fp16",
            torch.randn(3, 5, generator=generator, dtype=torch.float16),
            torch.randn(6, 3, generator=generator, dtype=torch.float16),
            scale=0.75,
        ),
        _rank_lora(
            "linear_b",
            torch.randn(2, 3, generator=generator),
            torch.randn(4, 2, generator=generator),
        ),
        _rank_lora(
            "conv_3x3",
            torch.randn(2, 2, 3, 3, generator=generator),
            torch.randn(5, 2, 1, 1, generator=generator),
            multiplier=0.5,
        ),
    ]

    expected = [_compute_lora_effective_rank_stats(item) for item in loras]
    actual = _compute_lora_effective_rank_stats_batched(loras)

    assert [item["module"] for item in actual] == [item.lora_name for item in loras]
    for actual_item, expected_item in zip(actual, expected):
        _assert_stats_close(actual_item, expected_item)


def test_batched_rank_stats_analytic_scaling_zero_and_summary_schema():
    identity2 = torch.eye(2)
    identity3 = torch.eye(3)
    rank2 = _rank_lora("rank2", identity2, identity2, scale=2.0, multiplier=0.5)
    zero = _rank_lora("zero", identity2, torch.zeros_like(identity2), scale=3.0, multiplier=2.0)
    rank3 = _rank_lora("rank3", identity3, identity3)

    per_module = _compute_lora_effective_rank_stats_batched([rank2, zero, rank3])
    assert per_module[0]["energy"] == pytest.approx(2.0)
    assert per_module[0]["sat"] == pytest.approx(1.0)
    assert per_module[0]["top1"] == pytest.approx(0.5)
    assert per_module[1] == {"module": "zero", "r": 2, "sat": 0.0, "top1": 0.0, "energy": 0.0}
    assert per_module[2]["energy"] == pytest.approx(3.0)
    assert per_module[2]["sat"] == pytest.approx(1.0)
    assert per_module[2]["top1"] == pytest.approx(1.0 / 3.0)

    network = SimpleNamespace(unet_loras=[rank2, zero, rank3])
    summary = LoRANetwork.compute_rank_stats(network)
    assert summary["rank_dim"] is None
    assert summary["energy_sum"] == pytest.approx(5.0)
    assert list(summary["by_module"]) == ["rank2", "zero", "rank3"]
    assert summary["by_module"]["rank2"] is summary["per_module"][0]
    assert summary["sat_wmean"] == pytest.approx(1.0)
    assert summary["sat_p50"] == pytest.approx(1.0)
    assert summary["sat_p95"] == pytest.approx(1.0)
    assert summary["sat_max"] == pytest.approx(1.0)
    assert summary["top1_p95"] == pytest.approx(0.5 - (0.5 - 1.0 / 3.0) * 0.05)


def test_batched_rank_stats_preserve_cholesky_fallback(monkeypatch):
    regular = _rank_lora("regular", torch.eye(2), torch.eye(2))
    singular = _rank_lora(
        "singular",
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
    )

    expected = [
        _compute_lora_effective_rank_stats(regular),
        _compute_lora_effective_rank_stats(singular),
    ]
    scalar_calls = []
    original_scalar = _compute_lora_effective_rank_stats
    original_cholesky_ex = torch.linalg.cholesky_ex

    def tracked_scalar(item, eps=1e-12):
        scalar_calls.append(item.lora_name)
        return original_scalar(item, eps=eps)

    def force_second_batch_item_to_fallback(input, *args, **kwargs):
        chol, info = original_cholesky_ex(input, *args, **kwargs)
        if input.dim() == 3:
            info = info.clone()
            info[1] = 1
        return chol, info

    monkeypatch.setattr(lora_module, "_compute_lora_effective_rank_stats", tracked_scalar)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", force_second_batch_item_to_fallback)
    actual = _compute_lora_effective_rank_stats_batched([regular, singular])

    assert scalar_calls == ["singular"]
    for actual_item, expected_item in zip(actual, expected):
        _assert_stats_close(actual_item, expected_item)


def test_batched_rank_stats_skip_missing_modules_and_keep_inputs_and_rng_unchanged(monkeypatch):
    lora = _rank_lora("kept", torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0], [4.0]]))
    lora.lora_down.weight.grad = torch.tensor([[5.0, 6.0]])
    down_before = lora.lora_down.weight.detach().clone()
    up_before = lora.lora_up.weight.detach().clone()
    grad_before = lora.lora_down.weight.grad.clone()
    versions_before = (lora.lora_down.weight._version, lora.lora_up.weight._version)
    rng_before = torch.get_rng_state().clone()

    monkeypatch.setattr(lora_module, "_RANK_STATS_BATCH_MAX_BYTES", 1)
    result = _compute_lora_effective_rank_stats_batched(
        [
            SimpleNamespace(lora_name="missing"),
            lora,
            SimpleNamespace(lora_name="missing_up", lora_down=lora.lora_down),
        ]
    )

    assert [item["module"] for item in result] == ["kept"]
    assert torch.equal(lora.lora_down.weight, down_before)
    assert torch.equal(lora.lora_up.weight, up_before)
    assert torch.equal(lora.lora_down.weight.grad, grad_before)
    assert (lora.lora_down.weight._version, lora.lora_up.weight._version) == versions_before
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert LoRANetwork.compute_rank_stats(SimpleNamespace(unet_loras=[])) is None
    assert LoRANetwork.compute_rank_stats(SimpleNamespace(unet_loras=[lora]), scope="te") is None


def test_batched_rank_stats_fall_back_to_bounded_gram_chunks(monkeypatch):
    loras = [
        _rank_lora(f"rank2_{index}", torch.eye(2), torch.eye(2))
        for index in range(3)
    ]
    original_compute = lora_module._compute_rank_metrics_from_grams
    batch_sizes = []

    def tracked_compute(p, q, r, eps):
        batch_sizes.append(p.shape[0])
        return original_compute(p, q, r, eps)

    monkeypatch.setattr(lora_module, "_RANK_STATS_BATCH_MAX_BYTES", 1)
    monkeypatch.setattr(lora_module, "_compute_rank_metrics_from_grams", tracked_compute)

    result = _compute_lora_effective_rank_stats_batched(loras)

    assert [item["module"] for item in result] == [item.lora_name for item in loras]
    assert batch_sizes == [1, 1, 1]


def test_write_csv_rows_batches_an_event_and_appends_without_duplicate_header(
    tmp_path, monkeypatch
):
    path = tmp_path / "rank.csv"
    path.write_text("stale\n", encoding="utf-8")
    real_open = builtins.open
    open_modes = []

    def tracked_open(file, mode="r", *args, **kwargs):
        open_modes.append((str(file), mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracked_open)

    header_written = _write_csv_rows(str(path), "module,energy", ["a,1.000000", "b,2.000000"], False)
    assert header_written is True
    assert open_modes == [(str(path), "w")]
    assert path.read_text(encoding="utf-8") == "module,energy\na,1.000000\nb,2.000000\n"

    open_modes.clear()
    header_written = _write_csv_rows(str(path), "module,energy", ["c,3.000000"], header_written)
    assert header_written is True
    assert open_modes == [(str(path), "a")]
    assert path.read_text(encoding="utf-8") == (
        "module,energy\na,1.000000\nb,2.000000\nc,3.000000\n"
    )

    assert _write_csv_rows(str(path), "module,energy", [], header_written) is True
    assert path.read_text(encoding="utf-8").endswith("c,3.000000\n")
