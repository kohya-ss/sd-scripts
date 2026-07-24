import sys
import types

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


def _clone_linear(linear):
    cloned = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    cloned.load_state_dict(linear.state_dict())
    return cloned


def test_oftv2_linear_forward_matches_merge():
    torch.manual_seed(1)
    base = nn.Linear(8, 4, bias=False)
    base_for_merge = _clone_linear(base)
    x = torch.randn(3, 8)

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


def test_boft_linear_forward_matches_merge():
    torch.manual_seed(2)
    base = nn.Linear(8, 4, bias=False)
    base_for_merge = _clone_linear(base)
    x = torch.randn(3, 8)

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
