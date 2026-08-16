import math

import pytest

from library.generation_lora_strengths import (
    LoraComponentStrengths,
    apply_generation_strengths,
    component_strengths_from_spec,
    flatten_strength_specs,
    format_strength_spec,
    normalize_strength_spec,
    resolve_flat_strengths,
    serialize_strength_spec,
    validate_component_strength_compatibility,
)


class FakeModule:
    def __init__(self, multiplier=9.0, lbw_multiplier=0.25):
        self.multiplier = multiplier
        self.lbw_multiplier = lbw_multiplier


class FakeNetwork:
    def __init__(self):
        self.te1 = [FakeModule(), FakeModule()]
        self.te2 = [FakeModule()]
        self._text_encoder_loras_by_encoder = [self.te1, self.te2]
        self.text_encoder_loras = self.te1 + self.te2
        self.unet_loras = [FakeModule(), FakeModule()]
        self.multiplier = 9.0

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for module in self.text_encoder_loras + self.unet_loras:
            module.multiplier = multiplier


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.8, (0.8,)),
        ([1.0, 0.5], (1.0, 0.5)),
        ((1.0, 0.25, 0.5), (1.0, 0.25, 0.5)),
        ("1, .25, .5", (1.0, 0.25, 0.5)),
    ],
)
def test_normalize_strength_spec(value, expected):
    assert normalize_strength_spec(value) == expected


@pytest.mark.parametrize("value", [[], [1, 2, 3, 4], "1,,2", [math.nan], [math.inf]])
def test_normalize_strength_spec_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        normalize_strength_spec(value)


def test_format_and_serialize_strength_spec():
    assert format_strength_spec((1.0, 0.25, 0.5)) == "1, 0.25, 0.5"
    assert serialize_strength_spec((0.8,)) == 0.8
    assert serialize_strength_spec((1.0, 0.5)) == [1.0, 0.5]


def test_component_strengths_from_each_spec_shape():
    assert component_strengths_from_spec((0.8,)) == LoraComponentStrengths(0.8, 0.8, 0.8, 1)
    assert component_strengths_from_spec((1.0, 0.5)) == LoraComponentStrengths(1.0, 1.0, 0.5, 2)
    assert component_strengths_from_spec((1.0, 0.25, 0.5)) == LoraComponentStrengths(1.0, 0.25, 0.5, 3)


def test_resolve_one_network_uses_one_two_or_three_values():
    assert resolve_flat_strengths([0.8], 1) == (LoraComponentStrengths(0.8, 0.8, 0.8, 1),)
    assert resolve_flat_strengths([1.0, 0.5], 1) == (LoraComponentStrengths(1.0, 1.0, 0.5, 2),)
    assert resolve_flat_strengths([1.0, 0.25, 0.5], 1) == (
        LoraComponentStrengths(1.0, 0.25, 0.5, 3),
    )


def test_resolve_two_networks_groups_values_per_network():
    assert resolve_flat_strengths([0.8, 0.6], 2) == (
        LoraComponentStrengths(0.8, 0.8, 0.8, 1),
        LoraComponentStrengths(0.6, 0.6, 0.6, 1),
    )
    assert resolve_flat_strengths([1.0, 0.5, 0.8, 0.4], 2) == (
        LoraComponentStrengths(1.0, 1.0, 0.5, 2),
        LoraComponentStrengths(0.8, 0.8, 0.4, 2),
    )
    assert resolve_flat_strengths([1.0, 0.25, 0.5, 0.8, 1.0, 0.4], 2) == (
        LoraComponentStrengths(1.0, 0.25, 0.5, 3),
        LoraComponentStrengths(0.8, 1.0, 0.4, 3),
    )


def test_legacy_missing_cli_values_default_to_one():
    assert resolve_flat_strengths([0.8], 3) == (
        LoraComponentStrengths(0.8, 0.8, 0.8, 1),
        LoraComponentStrengths(1.0, 1.0, 1.0, 1),
        LoraComponentStrengths(1.0, 1.0, 1.0, 1),
    )


def test_legacy_missing_prompt_values_repeat_last():
    assert resolve_flat_strengths([0.8, 0.5], 3, repeat_last_legacy_value=True) == (
        LoraComponentStrengths(0.8, 0.8, 0.8, 1),
        LoraComponentStrengths(0.5, 0.5, 0.5, 1),
        LoraComponentStrengths(0.5, 0.5, 0.5, 1),
    )


def test_resolve_rejects_ambiguous_count():
    with pytest.raises(ValueError, match="expected at most 2 common strengths"):
        resolve_flat_strengths([1, 2, 3, 4, 5], 2)


def test_component_strength_compatibility_rejects_static_weight_modes():
    common = resolve_flat_strengths([0.8], 1)
    split = resolve_flat_strengths([1.0, 0.5], 1)

    validate_component_strength_compatibility(common, network_merge=True, network_pre_calc=True)
    with pytest.raises(ValueError, match="weight merging"):
        validate_component_strength_compatibility(split, network_merge=True)
    with pytest.raises(ValueError, match="network_pre_calc"):
        validate_component_strength_compatibility(split, network_pre_calc=True)


def test_partial_merge_rechecks_strengths_against_runtime_network_count():
    # Two values are legacy common strengths while two networks exist, but
    # become a TE/U-Net split after one network has been merged away.
    original = resolve_flat_strengths([0.5, 0.8], 2, repeat_last_legacy_value=True)
    remaining = resolve_flat_strengths([0.5, 0.8], 1, repeat_last_legacy_value=True)

    validate_component_strength_compatibility(original, network_merge=True)
    with pytest.raises(ValueError, match="weight merging"):
        validate_component_strength_compatibility(remaining, network_merge=True)


def test_flatten_specs_uses_widest_mode_and_expands_other_slots():
    assert flatten_strength_specs([(0.8,), (1.0, 0.5)]) == (0.8, 0.8, 1.0, 0.5)
    assert flatten_strength_specs([(0.8,), (1.0, 0.5), (1.0, 0.25, 0.5)]) == (
        0.8,
        0.8,
        0.8,
        1.0,
        1.0,
        0.5,
        1.0,
        0.25,
        0.5,
    )


@pytest.mark.parametrize(
    ("strengths", "expected_te1", "expected_te2", "expected_unet"),
    [
        (LoraComponentStrengths(0.8, 0.8, 0.8, 1), 0.8, 0.8, 0.8),
        (LoraComponentStrengths(1.0, 1.0, 0.5, 2), 1.0, 1.0, 0.5),
        (LoraComponentStrengths(1.0, 0.25, 0.5, 3), 1.0, 0.25, 0.5),
    ],
)
def test_apply_generation_strengths_by_scope(strengths, expected_te1, expected_te2, expected_unet):
    network = FakeNetwork()
    original_lbw = [module.lbw_multiplier for module in network.text_encoder_loras + network.unet_loras]

    apply_generation_strengths(network, strengths)

    assert {module.multiplier for module in network.te1} == {expected_te1}
    assert {module.multiplier for module in network.te2} == {expected_te2}
    assert {module.multiplier for module in network.unet_loras} == {expected_unet}
    assert [module.lbw_multiplier for module in network.text_encoder_loras + network.unet_loras] == original_lbw


def test_apply_split_rejects_unsupported_network():
    class Unsupported:
        def set_multiplier(self, multiplier):
            self.multiplier = multiplier

    with pytest.raises(ValueError, match="does not expose"):
        apply_generation_strengths(Unsupported(), LoraComponentStrengths(1.0, 1.0, 0.5, 2))
