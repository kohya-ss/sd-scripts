import copy
import sys
import types

import pytest
import torch
from torch import nn

try:
    import library.utils
except ModuleNotFoundError as e:
    if e.name != "diffusers":
        raise
    utils_mod = types.ModuleType("library.utils")
    utils_mod.setup_logging = lambda *args, **kwargs: None
    sys.modules["library.utils"] = utils_mod

from networks import boft, oft_v2


class Transformer2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.to_q(x)


class ToyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Transformer2DModel()

    def forward(self, x):
        return self.block(x)


class CLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.to_q(x)


class ToyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = CLIPAttention()

    def forward(self, x):
        return self.block(x)


def _base_and_input(layer):
    if layer == "linear":
        return nn.Linear(8, 4, bias=False), torch.randn(3, 8)
    kernel_size = 1 if layer == "conv1x1" else 3
    return nn.Conv2d(8, 4, kernel_size, padding=kernel_size // 2, bias=True), torch.randn(2, 8, 5, 5)


@pytest.mark.parametrize("layer", ["linear", "conv1x1", "conv3x3"])
def test_oftv2_linear_forward_matches_merge(layer):
    torch.manual_seed(1)
    base, x = _base_and_input(layer)
    base_for_merge = copy.deepcopy(base)

    module = oft_v2.OFTv2Module("lora_unet_block_to_q", "lora_unet", "block.to_q", base, block_size=4)
    module.oft_R.weight.data.normal_(0, 0.05)
    module.apply_to()

    merge_module = oft_v2.OFTv2Module("lora_unet_block_to_q", "lora_unet", "block.to_q", base_for_merge, block_size=4)
    merge_module.load_state_dict(module.state_dict(), strict=False)
    merge_module.merge_to()

    with torch.no_grad():
        hooked = base(x)
        merged = base_for_merge(x)

    assert torch.allclose(hooked, merged, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("layer", ["linear", "conv1x1", "conv3x3"])
def test_boft_linear_forward_matches_merge(layer):
    torch.manual_seed(2)
    base, x = _base_and_input(layer)
    base_for_merge = copy.deepcopy(base)

    module = boft.BOFTModule("lora_unet_block_to_q", "lora_unet", "block.to_q", base, block_size=4)
    module.boft_R.data.normal_(0, 0.04)
    module.boft_s.data.normal_(1.0, 0.02)
    module.apply_to()

    merge_module = boft.BOFTModule("lora_unet_block_to_q", "lora_unet", "block.to_q", base_for_merge, block_size=4)
    merge_module.load_state_dict(module.state_dict(), strict=False)
    merge_module.merge_to()

    with torch.no_grad():
        hooked = base(x)
        merged = base_for_merge(x)

    assert torch.allclose(hooked, merged, atol=1e-5, rtol=1e-5)


def test_oftv2_create_network_from_peft_style_weights():
    toy = ToyUNet()
    weights_sd = {
        "base_model.model.block.to_q.oft_R.weight": torch.zeros(2, 6),
    }

    network, returned_sd = oft_v2.create_network_from_weights(1.0, None, None, [], toy, weights_sd=weights_sd)

    assert returned_sd is weights_sd
    assert len(network.unet_loras) == 1
    assert network.unet_loras[0].lora_name == "lora_unet_block_to_q"
    assert network.unet_loras[0].oft_R.weight.shape == (2, 6)


def test_boft_create_network_from_peft_style_weights():
    toy = ToyUNet()
    weights_sd = {
        "base_model.model.block.to_q.boft_R": torch.zeros(1, 2, 4, 4),
        "base_model.model.block.to_q.boft_s": torch.ones(4, 1),
    }

    network, returned_sd = boft.create_network_from_weights(1.0, None, None, [], toy, weights_sd=weights_sd)

    assert returned_sd is weights_sd
    assert len(network.unet_loras) == 1
    assert network.unet_loras[0].lora_name == "lora_unet_block_to_q"
    assert network.unet_loras[0].boft_R.shape == (1, 2, 4, 4)
    assert network.unet_loras[0].boft_s.shape == (4, 1)


def test_boft_training_load_accepts_peft_style_weights(tmp_path):
    toy = ToyUNet()
    weights_sd = {
        "base_model.model.block.to_q.boft_R": torch.randn(1, 2, 4, 4) * 0.01,
        "base_model.model.block.to_q.boft_s": torch.ones(4, 1),
    }
    weights_file = tmp_path / "boft.pt"
    torch.save(weights_sd, weights_file)

    network = boft.create_network(1.0, 4, None, None, [], toy)
    network.apply_to([], toy, apply_text_encoder=False, apply_unet=True)
    network.load_weights(str(weights_file))

    assert torch.allclose(network.unet_loras[0].boft_R, weights_sd["base_model.model.block.to_q.boft_R"])


@pytest.mark.parametrize("block_share", [False, True])
@pytest.mark.parametrize("use_cayley_neumann", [False, True], ids=["exact", "neumann"])
@pytest.mark.parametrize("initial_scale", [0.0, 0.2], ids=["zero", "projected"])
def test_coft_backward_updates_rotation(initial_scale, use_cayley_neumann, block_share):
    torch.manual_seed(13)
    base = nn.Linear(8, 4, bias=False).requires_grad_(False)
    base_before = base.weight.detach().clone()
    module = oft_v2.OFTv2Module(
        "lora_unet_block_to_q", "lora_unet", "block.to_q", base,
        block_size=4, coft=True, coft_eps=0.05, block_share=block_share,
    )
    module.oft_R.use_cayley_neumann = use_cayley_neumann
    with torch.no_grad():
        module.oft_R.weight.normal_(0, initial_scale)
    initial = module.oft_R.weight.detach().clone()
    module.apply_to()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    # 修正前にも backward 自体は実行し、アダプターへの勾配欠落を検出する。
    x = torch.randn(3, 8, requires_grad=True)
    target = torch.randn(3, 4)
    loss = (base(x) - target).square().mean()
    projected = module.oft_R.weight.detach().clone()
    assert torch.isfinite(projected).all()
    if initial_scale:
        assert not torch.equal(initial, projected)
    else:
        assert torch.equal(initial, projected)
    loss.backward()

    grad = module.oft_R.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
    optimizer.step()
    assert not torch.equal(module.oft_R.weight.detach(), projected)
    assert base.weight.grad is None
    assert torch.equal(base.weight.detach(), base_before)


@pytest.mark.parametrize("backend", [oft_v2, boft], ids=["oftv2", "boft"])
@pytest.mark.parametrize("encoder_count", [1, 2], ids=["sd1", "sdxl"])
@pytest.mark.parametrize(
    "te_lrs, expected_lrs",
    [
        ([0.001, 0.002], [0.001, 0.002]),
        ([0.001], [0.001, 0.001]),
        (0.001, [0.001, 0.001]),
        (None, [0.003, 0.003]),
        ([], [0.003, 0.003]),
        ([0.0, 0.002], [0.0, 0.002]),
        ([0.001, 0.0], [0.001, 0.0]),
    ],
    ids=["distinct", "single-list", "scalar", "default", "empty", "te1-zero", "te2-zero"],
)
def test_text_encoder_optimizer_groups(backend, encoder_count, te_lrs, expected_lrs):
    encoders = [ToyTextEncoder() for _ in range(encoder_count)]
    unet = ToyUNet()
    for model in [*encoders, unet]:
        model.requires_grad_(False)
    network = backend.create_network(1.0, 4, None, None, encoders, unet)
    network.apply_to(encoders, unet)
    assert len(network.text_encoder_loras) == encoder_count
    assert len(network.unet_loras) == 1
    groups, descriptions = network.prepare_optimizer_params_with_multiple_te_lrs(te_lrs, 0.004, 0.003)
    # 実際の登録済みパラメーターで、グループ間の重複も検出する。
    optimizer = torch.optim.AdamW(groups, lr=0.003)
    parameter_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert len(descriptions) == len(groups)
    registered_ids = {id(p) for p in network.parameters()}
    assert set(parameter_ids) <= registered_ids
    actual_lrs = {id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]}
    expected = {}
    for index, lora in enumerate(network.text_encoder_loras):
        expected_prefix = "lora_te" if encoder_count == 1 else f"lora_te{index + 1}"
        assert lora.root_prefix == expected_prefix
        if expected_lrs[index] != 0:
            expected.update({id(p): expected_lrs[index] for p in lora.parameters()})
    for lora in network.unet_loras:
        expected.update({id(p): 0.004 for p in lora.parameters()})
    assert actual_lrs == expected


def _local_adapter_state(backend):
    if backend is oft_v2:
        return {"oft_R.weight": torch.full((2, 6), 0.025)}
    return {
        "boft_R": torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) * 0.001,
        "boft_s": torch.full((4, 1), 1.125),
    }


@pytest.mark.parametrize("backend", [oft_v2, boft], ids=["oftv2", "boft"])
@pytest.mark.parametrize(
    "source_prefix, target_index",
    [("clip_l.block.to_q", 0), ("clip_g.block.to_q", 1),
     ("lora_te1_block_to_q", 0), ("lora_te2_block_to_q", 1)],
    ids=["clip-l", "clip-g", "native-te1", "native-te2"],
)
def test_encoder_weight_namespace_isolation(backend, source_prefix, target_index, tmp_path):
    local_state = _local_adapter_state(backend)
    weights_sd = {f"{source_prefix}.{name}": value for name, value in local_state.items()}
    encoders = [ToyTextEncoder(), ToyTextEncoder()]
    network, returned_sd = backend.create_network_from_weights(
        1.0, None, None, encoders, ToyUNet(), weights_sd=weights_sd,
    )
    assert returned_sd is weights_sd
    assert [lora.root_prefix for lora in network.text_encoder_loras] == [f"lora_te{target_index + 1}"]
    assert network.unet_loras == []
    for name, value in local_state.items():
        assert torch.equal(network.text_encoder_loras[0].state_dict()[name], value)

    # 全ターゲットを作成済みの学習用ロードでも、他のエンコーダーを保持する。
    encoders = [ToyTextEncoder(), ToyTextEncoder()]
    unet = ToyUNet()
    training_network = backend.create_network(1.0, 4, None, None, encoders, unet)
    training_network.apply_to(encoders, unet)
    untouched = [training_network.text_encoder_loras[1 - target_index], *training_network.unet_loras]
    before = [{name: value.clone() for name, value in lora.state_dict().items()} for lora in untouched]
    weights_file = tmp_path / "adapter.pt"
    torch.save(weights_sd, weights_file)
    training_network.load_weights(str(weights_file))
    for name, value in local_state.items():
        assert torch.equal(training_network.text_encoder_loras[target_index].state_dict()[name], value)
    for lora, previous in zip(untouched, before):
        for name, value in previous.items():
            assert torch.equal(lora.state_dict()[name], value)


@pytest.mark.parametrize("backend", [oft_v2, boft], ids=["oftv2", "boft"])
def test_nested_peft_module_prefix_is_supported(backend):
    local_state = _local_adapter_state(backend)
    weights_sd = {f"base_model.model.encoder.block.to_q.{name}": value for name, value in local_state.items()}
    network, _ = backend.create_network_from_weights(1.0, None, None, [], ToyUNet(), weights_sd=weights_sd)
    assert len(network.unet_loras) == 1
    for name, value in local_state.items():
        assert torch.equal(network.unet_loras[0].state_dict()[name], value)


@pytest.mark.parametrize("backend", [oft_v2, boft], ids=["oftv2", "boft"])
@pytest.mark.parametrize("reverse_order", [False, True], ids=["forward-order", "reverse-order"])
def test_ambiguous_peft_suffix_is_rejected(backend, reverse_order):
    local_state = _local_adapter_state(backend)
    entries = [
        (f"base_model.model.{encoder}.block.to_q.{name}", value + index)
        for index, encoder in enumerate(["encoder_a", "encoder_b"])
        for name, value in local_state.items()
    ]
    if reverse_order:
        entries.reverse()
    with pytest.raises(ValueError, match="(?i)ambig"):
        backend.create_network_from_weights(1.0, None, None, [], ToyUNet(), weights_sd=dict(entries))
