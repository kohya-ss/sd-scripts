from __future__ import annotations

import copy

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_adamw8bitfast_state_dict_restore_on_real_gpu():
    pytest.importorskip("bitsandbytes")
    from library.adamw8bit_fast import AdamW8bitFast

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    parameter = torch.nn.Parameter(torch.randn(128, device="cuda"))
    optimizer = AdamW8bitFast([parameter], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (parameter.square().mean()).backward()
        optimizer.step()
    weight_state = parameter.detach().clone()
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    optimizer.zero_grad(set_to_none=True)
    (parameter.square().mean()).backward()
    optimizer.step()
    expected = parameter.detach().clone()

    with torch.no_grad():
        parameter.copy_(weight_state)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    optimizer.zero_grad(set_to_none=True)
    (parameter.square().mean()).backward()
    optimizer.step()
    torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_adamw8bitfast_full_prefix_snapshot_restores_all_mutable_state_on_real_gpu():
    pytest.importorskip("bitsandbytes")
    from types import SimpleNamespace

    from dq_profile.copied_train_network import GradNormGuardian, GradNormGuardianConfig
    from dq_profile.snapshot import TrainingSnapshot
    from dq_profile.v2_calibration import capture_state_bundle, compare_numeric_trees, fingerprint_tree
    from library.adamw8bit_fast import AdamW8bitFast

    torch.manual_seed(321)
    torch.cuda.manual_seed_all(321)
    network = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(128, 64, bias=False),
    ).cuda()
    network.train()
    optimizer = AdamW8bitFast(network.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 / (1.0 + 0.01 * step),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    guardian = GradNormGuardian(
        GradNormGuardianConfig(
            skip_grad_norm=False,
            log_grad_norm=False,
            log_grad_scale=False,
            log_grad_cosine=True,
            skip_grad_norm_max=None,
            nan_to_window=False,
            inf_to_window=False,
            skip_nan_immediate=True,
            skip_inf_immediate=True,
            moving_avg_window=8,
        )
    )
    trainer = SimpleNamespace(
        _te_lr_after_cfg={"enabled": True, "step": 20},
        _te_lr_after_resume_state={"applied": False},
        _te_lr_after_resumed=False,
        _te_lr_after_resume_step=7,
        _te_freeze_cfg={0: {"step": 10}},
        _te_frozen_state_dict={"marker": torch.tensor([1.0])},
        _te_frozen_param_ids=set(),
    )
    inputs = torch.linspace(-1.0, 1.0, 16 * 128, device="cuda").reshape(16, 128)

    def advance(step: int) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = network(inputs).square().mean()
        scaler.scale(loss).backward()
        assert guardian.observe(network, epoch=0, step=step, loss_val=float(loss.detach())) is False
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    for step in range(3):
        advance(step)
    snapshot = TrainingSnapshot.capture(
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer=trainer,
        guardian=guardian,
        global_step=3,
        epoch=0,
        data_step=3,
    )
    snapshot_bundle = capture_state_bundle(
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer=trainer,
        guardian=guardian,
    )

    for step in range(3, 6):
        advance(step)
    expected_bundle = capture_state_bundle(
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer=trainer,
        guardian=guardian,
    )

    trainer._te_lr_after_resumed = True
    trainer._te_lr_after_resume_step = 999
    snapshot.restore(
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer=trainer,
        guardian=guardian,
    )
    restored_bundle = capture_state_bundle(
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer=trainer,
        guardian=guardian,
    )
    assert restored_bundle["fingerprints"] == snapshot_bundle["fingerprints"]

    for step in range(3, 6):
        advance(step)
    actual_bundle = capture_state_bundle(
        network=network,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        trainer=trainer,
        guardian=guardian,
    )

    for component in ("network", "optimizer"):
        comparison = compare_numeric_trees(
            expected_bundle["numeric"][component],
            actual_bundle["numeric"][component],
        )
        assert comparison["topology_matches"] is True
        assert comparison["max_abs"] <= 1e-7
        assert comparison["relative_l2"] <= 1e-6
    for component in ("network_runtime", "scheduler", "scaler", "guardian", "trainer"):
        assert expected_bundle["fingerprints"][component] == actual_bundle["fingerprints"][component]
    assert fingerprint_tree(torch.cuda.get_rng_state_all())