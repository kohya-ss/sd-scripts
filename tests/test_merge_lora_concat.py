import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MERGERS = ["merge_lora", "sdxl_merge_lora", "flux_merge_lora"]
MODULE = "lora_unet_middle_block_1_transformer_blocks_0_attn1_to_q"
INPUT_METADATA = {"ss_v2": "False", "ss_base_model_version": "test-base"}


@pytest.fixture(params=MERGERS, ids=["sd", "sdxl", "flux"])
def merger(request, monkeypatch):
    # These command-line scripts import sibling modules by their bare names.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "networks"))
    return importlib.import_module(request.param)


def _weights(down, up, alpha, name=MODULE):
    weights = {f"{name}.lora_down.weight": down.clone(), f"{name}.lora_up.weight": up.clone()}
    if alpha is not None:
        weights[f"{name}.alpha"] = torch.tensor(float(alpha))
    return weights


def _factors(rank, alpha, *, name=MODULE, kernel=None, offset=0):
    down_shape = (rank, 3) if kernel is None else (rank, 3, kernel, kernel)
    up_shape = (2, rank) if kernel is None else (2, rank, 1, 1)
    down = torch.arange(torch.Size(down_shape).numel(), dtype=torch.float32).reshape(down_shape) / 17 - 0.5 + offset
    up = torch.arange(torch.Size(up_shape).numel(), dtype=torch.float32).reshape(up_shape) / 11 - 0.25 - offset
    return _weights(down, up, alpha, name)


def _write_models(directory, *states):
    paths = []
    for index, state in enumerate(states):
        path = directory / f"input-{index}.safetensors"
        save_file(state, str(path), metadata=INPUT_METADATA)
        paths.append(str(path))
    return paths


def _merge(merger, paths, ratios, *, concat=True, shuffle=False, lbws=None):
    options = {"concat": concat, "shuffle": shuffle}
    if merger.__name__ == "sdxl_merge_lora":
        result = merger.merge_lora_models(paths, ratios, lbws or [], torch.float32, **options)
    else:
        result = merger.merge_lora_models(paths, ratios, torch.float32, **options)
    return result[:2]


def _effective_weights(state):
    effective = {}
    for key, down in state.items():
        if not key.endswith(".lora_down.weight"):
            continue
        name = key.removesuffix(".lora_down.weight")
        up = state[f"{name}.lora_up.weight"].float()
        down = down.float()
        product = up @ down if down.ndim == 2 else torch.einsum("or,rihw->oihw", up[:, :, 0, 0], down)
        effective[name] = product * (state[f"{name}.alpha"].float() / down.shape[0])
    return effective


def _single_input_sum(merger, paths, ratios, lbws=None):
    expected = {}
    for index, (path, ratio) in enumerate(zip(paths, ratios)):
        state, _ = _merge(merger, [path], [ratio], concat=False, lbws=None if lbws is None else [lbws[index]])
        for name, update in _effective_weights(state).items():
            expected[name] = expected.get(name, 0) + update
    return expected


def _assert_effective(state, expected):
    actual = _effective_weights(state)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=2e-6)


def _roundtrip(merger, path, state, metadata, dtype):
    state = dict(state)
    if merger.__name__ == "merge_lora":
        merger.save_to_file(str(path), state, state, dtype, metadata)
    elif merger.__name__ == "sdxl_merge_lora":
        merger.save_to_file(str(path), {key: value.to(dtype) for key, value in state.items()}, metadata)
    else:
        merger.save_to_file(str(path), state, dtype, metadata)
    with safe_open(str(path), framework="pt") as saved:
        assert saved.metadata() == metadata
    return load_file(str(path))


def _consumer_forward(merger, state, inputs, kernel):
    consumer_name = "lora_flux" if merger.__name__ == "flux_merge_lora" else "lora"
    consumer = importlib.import_module(consumer_name).LoRAModule
    with torch.random.fork_rng():
        base = torch.nn.Linear(3, 2) if kernel is None else torch.nn.Conv2d(3, 2, kernel, padding=kernel // 2)
        with torch.no_grad():
            base.weight.fill_(0.125)
            base.bias.fill_(0.375)
            base_output = base(inputs)
        adapter = consumer(
            MODULE, base, lora_dim=state[f"{MODULE}.lora_down.weight"].shape[0], alpha=state[f"{MODULE}.alpha"]
        )
        adapter.apply_to()
        adapter.load_state_dict({key.removeprefix(MODULE + "."): value.float() for key, value in state.items()})
        adapter.eval()
        with torch.no_grad():
            return base_output, base(inputs)


@pytest.mark.parametrize("second_ratio", [1, 0], ids=["sum", "zero-contribution"])
def test_concat_preserves_explicit_effective_sum_and_output_metadata(tmp_path, merger, second_ratio):
    update = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state = _weights(torch.eye(2), update, 2)
    paths = _write_models(tmp_path, state, state)
    merged, metadata = _merge(merger, paths, [1, second_ratio])

    _assert_effective(merged, {MODULE: (1 + second_ratio) * update})
    _assert_effective(merged, _single_input_sum(merger, paths, [1, second_ratio]))
    assert merged[f"{MODULE}.lora_down.weight"].shape[0] == 4
    assert merged[f"{MODULE}.alpha"].item() == 4
    assert metadata["ss_network_dim"] == "4"
    assert float(metadata["ss_network_alpha"]) == 4


@pytest.mark.parametrize(
    "alphas,ratios",
    [((1, 6), (0.5, -1.25)), ((None, None), (1, 1)), ((-1, -6), (-0.5, 1.25))],
    ids=["different-alpha-signed-ratios", "missing-alpha", "negative-alpha"],
)
def test_unequal_ranks_preserve_independently_processed_inputs(tmp_path, merger, alphas, ratios):
    paths = _write_models(tmp_path, _factors(2, alphas[0]), _factors(3, alphas[1], offset=0.2))
    merged, _ = _merge(merger, paths, ratios)
    expected = _single_input_sum(merger, paths, ratios)

    # The existing convention is q * abs(q), including in ordinary merging.
    raw = [_effective_weights(_merge(merger, [path], [1], concat=False)[0])[MODULE] for path in paths]
    torch.testing.assert_close(expected[MODULE], sum(q * abs(q) * weight for q, weight in zip(ratios, raw)))
    _assert_effective(merged, expected)


def test_rank_and_alpha_metadata_are_computed_per_output_module(tmp_path, merger):
    first_only, later_only = "lora_unet_first_only", "lora_unet_later_only"
    first = {**_factors(2, 2), **_factors(2, 2, name=first_only, offset=0.3)}
    second = {**_factors(3, 6, offset=0.1), **_factors(2, 2, name=later_only, offset=0.6)}
    paths = _write_models(tmp_path, first, second)
    merged, metadata = _merge(merger, paths, [1, 1])

    _assert_effective(merged, _single_input_sum(merger, paths, [1, 1]))
    assert {name: merged[f"{name}.lora_down.weight"].shape[0] for name in (MODULE, first_only, later_only)} == {
        MODULE: 5, first_only: 2, later_only: 2,
    }
    assert {name: merged[f"{name}.alpha"].item() for name in (MODULE, first_only, later_only)} == {
        MODULE: 5, first_only: 2, later_only: 2,
    }
    assert metadata["ss_network_dim"] == "Dynamic"
    assert metadata["ss_network_alpha"] == "Dynamic"


def test_shuffling_preserves_three_input_sum_without_changing_metadata(tmp_path, merger):
    paths = _write_models(tmp_path, _factors(2, 1), _factors(3, 4, offset=0.2), _factors(1, 3, offset=-0.1))
    ratios = [-0.5, 1.5, 0.25]
    ordinary, metadata = _merge(merger, paths, ratios)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        shuffled, shuffled_metadata = _merge(merger, paths, ratios, shuffle=True)

    assert shuffled_metadata == metadata
    _assert_effective(shuffled, _effective_weights(ordinary))
    _assert_effective(shuffled, _single_input_sum(merger, paths, ratios))


@pytest.mark.parametrize(
    "merger,kernel,save_dtype",
    [
        ("merge_lora", None, torch.float32),
        ("merge_lora", 1, torch.float16),
        ("merge_lora", 3, torch.float32),
        ("sdxl_merge_lora", None, torch.float32),
        ("sdxl_merge_lora", 3, torch.float16),
        ("flux_merge_lora", None, torch.bfloat16),
    ],
    indirect=["merger"],
)
def test_saved_concat_matches_actual_lora_module_outputs(tmp_path, merger, kernel, save_dtype):
    paths = _write_models(tmp_path, _factors(2, 1, kernel=kernel), _factors(3, 6, kernel=kernel, offset=0.1))
    ratios = [0.5, -1.25]
    merged, metadata = _merge(merger, paths, ratios)
    restored = _roundtrip(merger, tmp_path / "merged.safetensors", merged, metadata, save_dtype)
    input_shape = (2, 3) if kernel is None else (2, 3, 5, 5)
    inputs = torch.linspace(-1, 1, torch.Size(input_shape).numel()).reshape(input_shape)
    base_output, actual = _consumer_forward(merger, restored, inputs, kernel)
    expected = base_output.clone()
    assert torch.count_nonzero(base_output) > 0
    for path, ratio in zip(paths, ratios):
        single, _ = _merge(merger, [path], [ratio], concat=False)
        base, output = _consumer_forward(merger, single, inputs, kernel)
        expected += output - base

    tolerance = {torch.float32: 2e-5, torch.float16: 0.003, torch.bfloat16: 0.02}[save_dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert restored[f"{MODULE}.lora_down.weight"].dtype == save_dtype
    # Independently check the contracted matrix/convolution before quantization.
    _assert_effective(merged, _single_input_sum(merger, paths, ratios))


@pytest.mark.parametrize("merger", ["sdxl_merge_lora"], indirect=True)
@pytest.mark.parametrize("block_weight", [2, 0.5, 0, -0.5])
def test_sdxl_concat_preserves_block_weight_sign_and_mask_semantics(tmp_path, merger, block_weight):
    outside_mask = "lora_unet_input_blocks_1_0_op"
    update = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    first = {**_weights(torch.eye(2), update, 2), **_weights(torch.eye(2), update, 2, outside_mask)}
    second = {**_weights(torch.eye(2), 2 * update, 2), **_weights(torch.eye(2), 2 * update, 2, outside_mask)}
    paths = _write_models(tmp_path, first, second)
    lbws = [json.dumps([block_weight] * 12)] * 2
    ratios = [-0.5, 1]
    expected = _single_input_sum(merger, paths, ratios, lbws)
    torch.testing.assert_close(expected[MODULE], 1.75 * block_weight**2 * update)
    torch.testing.assert_close(expected[outside_mask], 1.75 * update)

    merged, _ = _merge(merger, paths, ratios, lbws=lbws)
    _assert_effective(merged, expected)


def test_nonconcat_factors_metadata_and_seeded_shuffle_remain_exact(tmp_path, merger):
    down1, up1 = torch.eye(2), torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    down2, up2 = torch.tensor([[2.0, -1.0], [1.0, 3.0]]), torch.tensor([[-2.0, 1.0], [4.0, -3.0]])
    paths = _write_models(tmp_path, _weights(down1, up1, 2), _weights(down2, up2, 8))
    if merger.__name__ == "sdxl_merge_lora":
        lbws = [json.dumps([value] * 12) for value in (0.5, -2)]
        expected_down, expected_up = 0.25 * down1 + down2, 0.25 * up1 - up2
    else:
        lbws = None
        expected_down, expected_up = 0.5 * down1 - 0.5 * down2, 0.5 * up1 + 0.5 * up2
    with torch.random.fork_rng():
        torch.manual_seed(7)
        merged, metadata = _merge(merger, paths, [0.5, -0.25], concat=False, shuffle=True, lbws=lbws)

    torch.testing.assert_close(merged[f"{MODULE}.lora_down.weight"], expected_down[[1, 0]], rtol=0, atol=0)
    torch.testing.assert_close(merged[f"{MODULE}.lora_up.weight"], expected_up[:, [1, 0]], rtol=0, atol=0)
    torch.testing.assert_close(merged[f"{MODULE}.alpha"], torch.tensor(2.0), rtol=0, atol=0)
    assert metadata == {
        **INPUT_METADATA, "ss_network_module": "networks.lora", "ss_network_dim": "2", "ss_network_alpha": "2.0"
    }
    paths = _write_models(tmp_path, _factors(2, 2), _factors(3, 3))
    with pytest.raises(AssertionError, match="weights shape mismatch"):
        _merge(merger, paths, [1, 1], concat=False)


@pytest.mark.parametrize("alpha", [None, 1], ids=["implicit-integer-alpha", "explicit-float-alpha"])
def test_single_input_and_disjoint_modules_keep_alpha_dtype_and_metadata(tmp_path, merger, alpha):
    first = _factors(2, alpha)
    later = _factors(2, alpha, name="lora_unet_later_only", offset=0.2)
    paths = _write_models(tmp_path, first, later)
    singles = [_merge(merger, [path], [1], concat=False) for path in paths]
    singleton, singleton_metadata = _merge(merger, paths[:1], [1])
    disjoint, metadata = _merge(merger, paths, [1, 1])

    for actual, expected in [(singleton, singles[0][0]), (disjoint, {**singles[0][0], **singles[1][0]})]:
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert singleton_metadata == metadata == singles[0][1] == singles[1][1]
    assert disjoint[f"{MODULE}.alpha"].dtype == (torch.int64 if alpha is None else torch.float32)
    assert metadata["ss_network_alpha"] == ("2" if alpha is None else "1.0")
