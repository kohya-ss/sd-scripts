import json
import math
from types import SimpleNamespace

import torch

from tools.make_lora_diagnostic_report import build_chart_payload, parse_grad_log, sanitize_json
import train_network
from train_network import (
    GradNormGuardian,
    GradNormGuardianConfig,
    NetworkTrainer,
    _calculate_grad_norm,
    _can_use_foreach_grad_norm,
    _legacy_grad_norm,
    resolve_avg_proxy_candidate_modes,
    resolve_grad_norm_settings,
)


def test_all_grad_norm_presets_disable_cosine_logging():
    for mode in ("stable", "stable_no_threshoff", "gamble"):
        settings = resolve_grad_norm_settings(SimpleNamespace(grad_norm_mode=mode))
        assert settings[3] is False


def test_fixed_avg_promote_scores_only_selected_mode():
    assert resolve_avg_proxy_candidate_modes("promote", "fixed", "ema") == ["ema"]
    assert resolve_avg_proxy_candidate_modes("promote", "fixed", "uniform") == ["uniform"]
    assert resolve_avg_proxy_candidate_modes("promote", "fixed", "metric") == ["metric"]


def test_best_avg_promote_keeps_comparison_candidates():
    assert resolve_avg_proxy_candidate_modes("promote", "best", "ema") == ["ema", "uniform"]
    assert resolve_avg_proxy_candidate_modes("promote", "best", "uniform") == ["ema", "uniform"]
    assert resolve_avg_proxy_candidate_modes("promote", "best", "metric") == ["ema", "uniform", "metric"]


def test_shadow_keeps_comparison_candidates():
    assert resolve_avg_proxy_candidate_modes("shadow", "fixed", "ema") == ["ema", "uniform"]
    assert resolve_avg_proxy_candidate_modes("shadow", "fixed", "uniform") == ["ema", "uniform"]
    assert resolve_avg_proxy_candidate_modes("shadow", "fixed", "metric") == ["ema", "uniform", "metric"]
    assert resolve_avg_proxy_candidate_modes("shadow", "best", "ema") == ["ema", "uniform"]


class _SingleProcessAccelerator:
    num_processes = 1

    def reduce(self, tensor, reduction):
        raise AssertionError("single-process training must not call accelerator.reduce")


class _MultiProcessAccelerator:
    num_processes = 2

    def __init__(self):
        self.reduce_calls = 0

    def reduce(self, tensor, reduction):
        assert reduction == "mean"
        self.reduce_calls += 1
        return tensor + 1


def _network_with_grad():
    network = torch.nn.Linear(2, 1, bias=False)
    network.weight.grad = torch.tensor([[2.0, 3.0]])
    return network


def test_all_reduce_network_is_noop_for_single_process():
    network = _network_with_grad()
    original_grad = network.weight.grad

    NetworkTrainer().all_reduce_network(_SingleProcessAccelerator(), network)

    assert network.weight.grad is original_grad
    assert torch.equal(network.weight.grad, torch.tensor([[2.0, 3.0]]))


def test_all_reduce_network_still_reduces_for_multiple_processes():
    network = _network_with_grad()
    accelerator = _MultiProcessAccelerator()

    NetworkTrainer().all_reduce_network(accelerator, network)

    assert accelerator.reduce_calls == 1
    assert torch.equal(network.weight.grad, torch.tensor([[3.0, 4.0]]))


def test_diagnostic_report_accepts_grad_log_without_cosine(tmp_path):
    log_path = tmp_path / "gradient_logs+without_cosine.txt"
    log_path.write_text(
        "Epoch,Step,Gradient Norm,Threshold,Loss,ThreshOff,Scale\n"
        "0,0,10.0,200000.0,0.5,0,65536\n"
        "0,1,12.0,200000.0,0.4,0,65536\n",
        encoding="utf-8",
    )

    grad_data = parse_grad_log(str(log_path), ma_window=2)
    charts = build_chart_payload(grad_data, None, None, None, None)

    assert grad_data["cosine"] == [None, None]
    assert grad_data["summary"]["cosine_valid_ratio"] == 0.0
    assert next(chart for chart in charts["grad"] if chart["id"] == "cosine")["series"][0]["y"] == [None, None]
    json.dumps(sanitize_json(charts), allow_nan=False)


def _guardian_config(**overrides):
    values = {
        "skip_grad_norm": False,
        "log_grad_norm": False,
        "log_grad_scale": False,
        "log_grad_cosine": False,
        "skip_grad_norm_max": None,
        "nan_to_window": False,
        "inf_to_window": False,
        "skip_nan_immediate": True,
        "skip_inf_immediate": True,
        "moving_avg_window": 200,
        "log_flush_interval": 100,
        "initial_threshold": 200_000.0,
    }
    values.update(overrides)
    return GradNormGuardianConfig(**values)


def _single_parameter_model(grad):
    model = torch.nn.Linear(grad.numel(), 1, bias=False, dtype=grad.dtype)
    model.weight.grad = grad.reshape_as(model.weight).clone()
    return model


def test_foreach_grad_norm_matches_legacy_and_does_not_modify_gradients(monkeypatch):
    grads = [
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        torch.tensor([[0.0, -12.0], [5.0, 0.0]], dtype=torch.float32),
        torch.zeros(7, dtype=torch.float32),
    ]
    snapshots = [grad.clone() for grad in grads]
    versions = [grad._version for grad in grads]
    expected = _legacy_grad_norm(grads)
    public_calls = []

    def fake_get_total_norm(current_grads, norm_type, error_if_nonfinite, foreach):
        public_calls.append((current_grads, norm_type, error_if_nonfinite, foreach))
        return expected.clone()

    def fail_if_legacy_is_used(_grads):
        raise AssertionError("supported public foreach path must not fall back to legacy")

    monkeypatch.setattr(train_network, "_FOREACH_GRAD_NORM_DISABLED", False)
    monkeypatch.setattr(train_network, "_TORCH_GET_TOTAL_NORM", fake_get_total_norm)
    monkeypatch.setattr(train_network, "_legacy_grad_norm", fail_if_legacy_is_used)
    actual = _calculate_grad_norm(grads)

    assert _can_use_foreach_grad_norm(grads)
    assert len(public_calls) == 1
    called_grads, norm_type, error_if_nonfinite, foreach = public_calls[0]
    assert called_grads is grads
    assert norm_type == 2.0
    assert error_if_nonfinite is False
    assert foreach is True
    assert train_network._FOREACH_GRAD_NORM_DISABLED is False
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    for grad, snapshot, version in zip(grads, snapshots, versions):
        assert grad._version == version
        torch.testing.assert_close(grad, snapshot, rtol=0.0, atol=0.0)


def test_grad_norm_handles_zero_and_no_gradients():
    zero_grad = torch.zeros(8, dtype=torch.float32)

    assert _calculate_grad_norm([zero_grad]).item() == 0.0
    assert _calculate_grad_norm([]).item() == 0.0

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = None
    guardian = GradNormGuardian(_guardian_config())

    assert guardian.observe(model, epoch=0, step=1, loss_val=0.5) is False
    assert list(guardian.moving_avg_window) == [0.0]


class _CountingGradModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(1))
        self.second = torch.nn.Parameter(torch.zeros(1))
        self.parameters_calls = 0

    def parameters(self, recurse=True):
        self.parameters_calls += 1
        return super().parameters(recurse=recurse)


def test_guardian_caches_parameters_while_active_gradients_change():
    model = _CountingGradModel()
    guardian = GradNormGuardian(_guardian_config())

    model.first.grad = torch.tensor([3.0])
    model.second.grad = None
    guardian.observe(model, epoch=0, step=1, loss_val=0.5)

    model.first.grad = None
    model.second.grad = torch.tensor([4.0])
    guardian.observe(model, epoch=0, step=2, loss_val=0.4)

    other_model = _CountingGradModel()
    other_model.first.grad = torch.tensor([5.0])
    guardian.observe(other_model, epoch=0, step=3, loss_val=0.3)

    assert model.parameters_calls == 1
    assert other_model.parameters_calls == 1
    assert list(guardian.moving_avg_window) == [3.0, 4.0, 5.0]


def test_unsupported_gradients_use_legacy_path(monkeypatch):
    gradient_sets = [
        [torch.tensor([3.0, 4.0], dtype=torch.float16)],
        [torch.tensor([3.0, 4.0], dtype=torch.bfloat16)],
        [
            torch.tensor([3.0], dtype=torch.float32),
            torch.tensor([4.0], dtype=torch.float64),
        ],
        [torch.tensor([3.0, 4.0], dtype=torch.float32).to_sparse()],
    ]

    def fail_if_foreach_is_used(_grads):
        raise AssertionError("unsupported gradients must use the legacy norm path")

    monkeypatch.setattr(train_network, "_foreach_grad_norm", fail_if_foreach_is_used)

    for grads in gradient_sets:
        assert not _can_use_foreach_grad_norm(grads)
        expected = _legacy_grad_norm(grads)
        actual = _calculate_grad_norm(grads)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    unsupported_device_grad = torch.empty(1, device="meta", dtype=torch.float32)
    legacy_result = torch.tensor(123.0)
    legacy_calls = 0

    def tracked_legacy(grads):
        nonlocal legacy_calls
        legacy_calls += 1
        assert len(grads) == 1
        assert grads[0] is unsupported_device_grad
        return legacy_result

    monkeypatch.setattr(train_network, "_legacy_grad_norm", tracked_legacy)

    assert not _can_use_foreach_grad_norm([unsupported_device_grad])
    assert _calculate_grad_norm([unsupported_device_grad]) is legacy_result
    assert legacy_calls == 1


def test_foreach_grad_norm_supports_older_torch_compatibility_path(monkeypatch):
    grads = [
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        torch.tensor([12.0], dtype=torch.float32),
    ]
    expected = _legacy_grad_norm(grads)
    original_foreach_norm = torch._foreach_norm
    foreach_calls = 0

    def tracked_foreach_norm(current_grads, norm_type):
        nonlocal foreach_calls
        foreach_calls += 1
        return original_foreach_norm(current_grads, norm_type)

    def fail_if_legacy_is_used(_grads):
        raise AssertionError("supported private foreach path must not fall back to legacy")

    monkeypatch.setattr(train_network, "_FOREACH_GRAD_NORM_DISABLED", False)
    monkeypatch.setattr(train_network, "_TORCH_GET_TOTAL_NORM", None)
    monkeypatch.setattr(torch, "_foreach_norm", tracked_foreach_norm)
    monkeypatch.setattr(train_network, "_legacy_grad_norm", fail_if_legacy_is_used)

    actual = _calculate_grad_norm(grads)

    assert foreach_calls == 1
    assert train_network._FOREACH_GRAD_NORM_DISABLED is False
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_foreach_unsupported_error_falls_back_once_without_hiding_other_errors(monkeypatch):
    grads = [torch.tensor([3.0, 4.0], dtype=torch.float32)]
    expected = _legacy_grad_norm(grads)
    foreach_calls = 0

    def unsupported_foreach(_grads):
        nonlocal foreach_calls
        foreach_calls += 1
        raise RuntimeError("foreach=True was passed, but this backend is not supported")

    monkeypatch.setattr(train_network, "_FOREACH_GRAD_NORM_DISABLED", False)
    monkeypatch.setattr(train_network, "_foreach_grad_norm", unsupported_foreach)

    torch.testing.assert_close(_calculate_grad_norm(grads), expected, rtol=0.0, atol=0.0)
    assert train_network._FOREACH_GRAD_NORM_DISABLED is True
    torch.testing.assert_close(_calculate_grad_norm(grads), expected, rtol=0.0, atol=0.0)
    assert foreach_calls == 1

    def unexpected_foreach(_grads):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(train_network, "_FOREACH_GRAD_NORM_DISABLED", False)
    monkeypatch.setattr(train_network, "_foreach_grad_norm", unexpected_foreach)

    try:
        _calculate_grad_norm(grads)
    except RuntimeError as error:
        assert str(error) == "CUDA out of memory"
    else:
        raise AssertionError("unexpected foreach errors must not be hidden")
    assert train_network._FOREACH_GRAD_NORM_DISABLED is False


def test_foreach_grad_norm_preserves_legacy_fp32_extreme_value_classification():
    large_grad = [torch.tensor([1e20], dtype=torch.float32)]
    tiny_grad = [torch.tensor([1e-30], dtype=torch.float32)]

    assert math.isinf(_legacy_grad_norm(large_grad).item())
    assert math.isinf(_calculate_grad_norm(large_grad).item())
    assert _legacy_grad_norm(tiny_grad).item() == 0.0
    assert _calculate_grad_norm(tiny_grad).item() == 0.0


def test_cosine_logging_keeps_legacy_observe_path_and_columns(monkeypatch, tmp_path):
    log_path = tmp_path / "gradient_logs+cosine.txt"
    model = _single_parameter_model(torch.tensor([3.0, 4.0]))

    def fail_if_fast_path_is_used(_grads):
        raise AssertionError("cosine diagnostics must keep the existing observe path")

    monkeypatch.setattr(train_network, "_calculate_grad_norm", fail_if_fast_path_is_used)
    guardian = GradNormGuardian(
        _guardian_config(log_grad_norm=True, log_grad_scale=True, log_grad_cosine=True),
        scaler_for_log=SimpleNamespace(get_scale=lambda: 1024.0),
        log_file_path=str(log_path),
    )

    guardian.observe(model, epoch=2, step=1, loss_val=0.25)
    guardian.observe(model, epoch=2, step=2, loss_val=0.20)

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "Epoch,Step,Gradient Norm,Threshold,Loss,ThreshOff,Scale,CosineSim"
    ]
    first_fields = guardian.log_buffer[0].strip().split(",")
    second_fields = guardian.log_buffer[1].strip().split(",")
    assert len(first_fields) == 8
    assert first_fields[-2] == "1024.0"
    assert math.isnan(float(first_fields[-1]))
    assert len(second_fields) == 8
    assert second_fields[-2] == "1024.0"
    assert math.isclose(float(second_fields[-1]), 1.0)
    assert list(guardian.moving_avg_window) == [5.0, 5.0]


def test_nonfinite_grad_norm_matches_legacy_classification_and_skip_behavior():
    cases = [
        (float("nan"), math.isnan),
        (float("inf"), math.isinf),
    ]

    for value, classifier in cases:
        grad = torch.tensor([value], dtype=torch.float32)
        assert classifier(_legacy_grad_norm([grad]).item())
        assert classifier(_calculate_grad_norm([grad]).item())

        model = _single_parameter_model(grad)
        guardian = GradNormGuardian(_guardian_config(skip_grad_norm=True))
        assert guardian.observe(model, epoch=0, step=1, loss_val=0.5) is True
        assert len(guardian.moving_avg_window) == 0


def test_guardian_threshold_cap_skip_decision_and_log_columns_are_unchanged(tmp_path):
    log_path = tmp_path / "gradient_logs+threshold.txt"
    model = _single_parameter_model(torch.tensor([3.0]))
    guardian = GradNormGuardian(
        _guardian_config(
            skip_grad_norm=True,
            log_grad_norm=True,
            skip_grad_norm_max=4.0,
            moving_avg_window=2,
            log_flush_interval=100,
            initial_threshold=100.0,
        ),
        log_file_path=str(log_path),
    )

    first_skip = guardian.observe(model, epoch=1, step=1, loss_val=0.5)
    model.weight.grad = torch.tensor([[5.0]])
    second_skip = guardian.observe(model, epoch=1, step=2, loss_val=0.4)

    assert first_skip is False
    assert second_skip is True
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "Epoch,Step,Gradient Norm,Threshold,Loss,ThreshOff"
    ]

    first_fields = guardian.log_buffer[0].strip().split(",")
    second_fields = guardian.log_buffer[1].strip().split(",")
    assert len(first_fields) == 6
    assert first_fields == ["1", "1", "3.0", "100.0", "0.5", "0"]
    assert len(second_fields) == 6
    assert second_fields == ["1", "2", "5.0", "4.0", "0.4", "0"]
