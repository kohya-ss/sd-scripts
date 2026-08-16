from pathlib import Path

import pytest

import sdxl_lora_report_cui as report_cui
import sdxl_lora_report_worker as worker


def item(path: str, strength, lbw="XLMLT1"):
    return {
        "name": Path(path).stem,
        "path": path,
        "strength": strength,
        "lbw": lbw,
        "module": "networks.lora_lbw",
    }


def test_cui_accepts_scalar_pair_and_triple_strengths(tmp_path):
    for raw, expected in ((0.8, 0.8), ([1.0, 0.5], [1.0, 0.5]), ([1.0, 0.25, 0.5], [1.0, 0.25, 0.5])):
        normalized = report_cui.normalize_lora_item(
            {"path": "test.safetensors", "strength": raw}, tmp_path, False, "condition", 1
        )
        assert normalized["strength"] == expected


def test_worker_one_slot_emits_one_two_or_three_am_values():
    slot = item("A.safetensors", 1.0)
    for strength, expected in (
        (0.8, "--am 0.8"),
        ([1.0, 0.5], "--am 1,0.5"),
        ([1.0, 0.25, 0.5], "--am 1,0.25,0.5"),
    ):
        job = {"prompt": "test", "seed": 1, "condition_items": [item("A.safetensors", strength)]}
        assert expected in worker.prompt_line(job, [slot])


def test_worker_expands_mixed_two_slot_modes_to_widest_mode():
    slots = [item("A.safetensors", 1.0), item("B.safetensors", 1.0)]
    job = {
        "prompt": "test",
        "seed": 1,
        "condition_items": [
            item("A.safetensors", [1.0, 0.5]),
            item("B.safetensors", [0.8, 1.0, 0.4]),
        ],
    }
    assert "--am 1,1,0.5,0.8,1,0.4" in worker.prompt_line(job, slots)


def test_worker_zeroes_inactive_slot_in_component_mode():
    slots = [item("A.safetensors", 1.0), item("B.safetensors", 1.0)]
    job = {
        "prompt": "test",
        "seed": 1,
        "condition_items": [item("A.safetensors", [1.0, 0.5])],
    }
    assert "--am 1,0.5,0,0" in worker.prompt_line(job, slots)


@pytest.mark.parametrize("strength", [[], [1, 2, 3, 4], [float("nan")]])
def test_cui_rejects_invalid_strength_specs(tmp_path, strength):
    with pytest.raises(ValueError):
        report_cui.normalize_lora_item(
            {"path": "test.safetensors", "strength": strength}, tmp_path, False, "condition", 1
        )
