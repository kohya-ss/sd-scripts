"""Tests for networks/lora_generic.py with the MOSS-Image architecture (small random DiT and Qwen3.5 text model)."""

import os

import pytest
import torch
from safetensors.torch import load_file

from library import moss_image_models, moss_image_utils
from networks import lora_generic, network_base

SMALL_CONFIG = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)
NUM_UNET_MODULES = 3 * 6  # q, k, v, o, ffn_up_gate, ffn_down per block
NUM_TEXT_EMBEDDER_MODULES = 1 + 2 * 5  # input projection + (qkv, proj, ffn_gate, ffn_up, ffn_down) per text block
NUM_TE_LAYERS = 4  # 3 linear attention layers + 1 full attention layer (layer_types of the config)
NUM_TE_MODULES = 3 * (3 + 3) + (4 + 3)  # GDN: in_proj_qkv, in_proj_z, out_proj (+ MLP 3), attention: q, k, v, o (+ MLP 3)


def make_dit(seed=0) -> moss_image_models.MossImage:
    torch.manual_seed(seed)
    return moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=False).eval()


def make_text_encoder(seed=0):
    from transformers import Qwen3_5TextModel

    config = moss_image_utils.load_text_encoder_config()
    config.num_hidden_layers = NUM_TE_LAYERS
    config.hidden_size = SMALL_CONFIG["text_dim"]
    config.intermediate_size = 48
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.head_dim = 12
    config.linear_num_value_heads = 2
    config.linear_num_key_heads = 1
    config.linear_key_head_dim = 8
    config.linear_value_head_dim = 8
    config.vocab_size = 256
    torch.manual_seed(seed)
    model = Qwen3_5TextModel(config)
    return model.eval()


def make_inputs(seed=1):
    torch.manual_seed(seed)
    batch, height, width = 2, 3, 5
    lens = [4, 7]
    x = torch.randn(batch, SMALL_CONFIG["latent_channels"], height, width)
    context = torch.randn(batch, max(lens), SMALL_CONFIG["text_dim"])
    mask = torch.zeros(batch, max(lens), dtype=torch.int32)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    context = context * mask[..., None]
    timesteps = torch.tensor([750.0, 120.0])
    return x, timesteps, context, mask


def test_detect_arch_config_moss_image():
    dit = make_dit()
    config = network_base.detect_arch_config(dit, [None])
    assert config.unet_target_modules == ["MossImageBlock"]
    assert config.adapter_target_modules == ["MossImageTextEmbedder"]
    assert config.te_prefixes == ["lora_te"]

    # text encoder only (e.g. inference scripts which load the text encoder separately)
    te = make_text_encoder()
    config = network_base.detect_arch_config(None, [te])
    assert config.unet_target_modules == ["MossImageBlock"]


def test_create_network_modules_and_names():
    dit = make_dit()
    te = make_text_encoder()
    network = lora_generic.create_network(1.0, 4, 2.0, None, [te], dit)
    assert isinstance(network, lora_generic.LoRANetwork)
    assert len(network.unet_loras) == NUM_UNET_MODULES
    assert len(network.text_encoder_loras) == NUM_TE_MODULES

    unet_names = {lora.lora_name for lora in network.unet_loras}
    assert "lora_unet_layers_0_proj_q" in unet_names
    assert "lora_unet_layers_2_proj_ffn_up_gate" in unet_names
    assert not any("text_embedder" in n or "modulation" in n or "final" in n for n in unet_names)

    te_names = {lora.lora_name for lora in network.text_encoder_loras}
    assert "lora_te_layers_0_linear_attn_in_proj_qkv" in te_names
    assert "lora_te_layers_0_linear_attn_out_proj" in te_names
    assert "lora_te_layers_3_self_attn_q_proj" in te_names
    assert "lora_te_layers_3_mlp_down_proj" in te_names
    assert not any("in_proj_a" in n or "in_proj_b" in n for n in te_names)  # default excludes

    lora = network.unet_loras[0]
    assert lora.lora_dim == 4 and float(lora.alpha) == 2.0 and lora.scale == 0.5


def test_train_text_embedder_alias():
    dit = make_dit()
    for key in ["train_text_embedder", "train_llm_adapter"]:
        network = lora_generic.create_network(1.0, 4, 1.0, None, [None], dit, **{key: "true"})
        assert len(network.unet_loras) == NUM_UNET_MODULES + NUM_TEXT_EMBEDDER_MODULES
        names = {lora.lora_name for lora in network.unet_loras}
        assert "lora_unet_text_embedder_blocks_0_qkv" in names
        assert "lora_unet_text_embedder_blocks_1_ffn_down" in names
        assert "lora_unet_text_embedder_proj" in names  # the input projection can be excluded with exclude_patterns

    network = lora_generic.create_network(
        1.0, 4, 1.0, None, [None], dit, train_text_embedder="true", exclude_patterns='["text_embedder\\\\.proj"]'
    )
    assert len(network.unet_loras) == NUM_UNET_MODULES + NUM_TEXT_EMBEDDER_MODULES - 1

    network = lora_generic.create_network(1.0, 4, 1.0, None, [None], dit)
    assert len(network.unet_loras) == NUM_UNET_MODULES


def test_forward_is_identity_at_init_and_backward_reaches_lora():
    dit = make_dit()
    te = make_text_encoder()
    x, timesteps, context, mask = make_inputs()
    with torch.no_grad():
        expected = dit(x, timesteps, context, mask)

    network = lora_generic.create_network(1.0, 4, 1.0, None, [te], dit)
    network.apply_to([te], dit, apply_text_encoder=True, apply_unet=True)
    network.prepare_grad_etc([te], dit)
    dit.requires_grad_(False)
    te.requires_grad_(False)

    # lora_up is zero-initialized: the output is unchanged
    with torch.no_grad():
        actual = dit(x, timesteps, context, mask)
    torch.testing.assert_close(actual, expected)

    # gradients flow to the LoRA parameters through the DiT and the text encoder
    network.train()
    ids = torch.randint(1, 256, (1, 6))
    hidden = te(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state  # [1, 6, 24]
    context2 = torch.cat([hidden, hidden.new_zeros(1, 1, hidden.shape[-1])], dim=1)
    mask2 = torch.tensor([[1, 1, 1, 1, 1, 1, 0]], dtype=torch.int32)
    out = dit(x[:1], timesteps[:1], context2, mask2)
    out.float().pow(2).mean().backward()

    for lora in network.unet_loras + network.text_encoder_loras:
        assert lora.lora_up.weight.grad is not None, lora.lora_name
        assert lora.lora_down.weight.grad is not None, lora.lora_name
    assert all(p.grad is None for p in dit.parameters())


def test_save_load_and_merge(tmp_path):
    dit = make_dit()
    x, timesteps, context, mask = make_inputs()

    network = lora_generic.create_network(1.0, 4, 1.0, None, [None], dit, train_text_embedder="true")
    network.apply_to([None], dit, apply_text_encoder=False, apply_unet=True)
    torch.manual_seed(2)
    for lora in network.unet_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=0.1)
    with torch.no_grad():
        expected = dit(x, timesteps, context, mask)

    path = os.path.join(tmp_path, "lora.safetensors")
    network.save_weights(path, torch.float32, {"ss_test": "1"})
    sd = load_file(path)
    assert "lora_unet_layers_0_proj_q.lora_down.weight" in sd
    assert "lora_unet_layers_0_proj_q.alpha" in sd
    assert "lora_unet_text_embedder_blocks_0_qkv.lora_up.weight" in sd

    # 1. training network re-created from the weights (adapter modules are picked up from the weights)
    dit2 = make_dit()
    network2, weights_sd = lora_generic.create_network_from_weights(1.0, path, None, [None], dit2)
    assert len(network2.unet_loras) == NUM_UNET_MODULES + NUM_TEXT_EMBEDDER_MODULES
    network2.apply_to([None], dit2, apply_text_encoder=False, apply_unet=True)
    info = network2.load_weights(path)
    assert len(info.missing_keys) == 0 and len(info.unexpected_keys) == 0
    with torch.no_grad():
        actual = dit2(x, timesteps, context, mask)
    torch.testing.assert_close(actual, expected)

    # 2. merge into the weights (inference modules)
    dit3 = make_dit()
    network3, weights_sd = lora_generic.create_network_from_weights(1.0, path, None, [None], dit3, for_inference=True)
    assert isinstance(network3.unet_loras[0], lora_generic.LoRAInfModule)
    network3.merge_to([None], dit3, weights_sd, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        merged = dit3(x, timesteps, context, mask)
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-5)

    # 3. the load-time merge hook (used by the inference scripts) gives the same weights
    from library.lora_utils import create_lora_merge_hook

    dit4 = make_dit()
    hook, unused_keys, _ = create_lora_merge_hook([dict(sd)], [1.0], torch.device("cpu"))
    merged_sd = {k: hook(k, v.clone()) for k, v in dit4.state_dict().items()}
    assert all(len(keys) == 0 for keys in unused_keys)
    dit4.load_state_dict(merged_sd)
    with torch.no_grad():
        merged = dit4(x, timesteps, context, mask)
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-5)


def test_text_encoder_lora_merge_hook_without_prefix():
    """Text encoder LoRA keys are matched without the `lora_te_` prefix by the load-time merge hook."""
    from library.lora_utils import create_lora_merge_hook

    te = make_text_encoder()
    dit = make_dit()
    network = lora_generic.create_network(1.0, 2, 1.0, None, [te], dit)
    network.apply_to([te], dit, apply_text_encoder=True, apply_unet=False)
    torch.manual_seed(3)
    for lora in network.text_encoder_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=0.1)
    ids = torch.randint(1, 256, (1, 5))
    with torch.no_grad():
        expected = te(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state

    lora_sd = {k[len("lora_te_") :]: v for k, v in network.state_dict().items() if k.startswith("lora_te_")}
    te2 = make_text_encoder()
    hook, unused_keys, _ = create_lora_merge_hook([lora_sd], None, torch.device("cpu"))
    merged_sd = {k: hook(k, v.clone()) for k, v in te2.state_dict().items()}
    assert all(len(keys) == 0 for keys in unused_keys)
    te2.load_state_dict(merged_sd)
    with torch.no_grad():
        merged = te2(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-5)


def test_optimizer_params_and_extras():
    dit = make_dit()
    te = make_text_encoder()
    network = lora_generic.create_network(
        1.0,
        4,
        1.0,
        None,
        [te],
        dit,
        network_reg_dims="layers\\.0\\..*=8",
        network_reg_lrs="layers\\.1\\..*=0.5",
        loraplus_unet_lr_ratio="4",
        rank_dropout="0.1",
        module_dropout="0.1",
        verbose="true",
    )
    network.apply_to([te], dit, apply_text_encoder=True, apply_unet=True)
    dims = {lora.lora_name: lora.lora_dim for lora in network.unet_loras}
    assert dims["lora_unet_layers_0_proj_q"] == 8 and dims["lora_unet_layers_1_proj_q"] == 4

    params, descriptions = network.prepare_optimizer_params_with_multiple_te_lrs([1e-4], 1e-3, 1e-3)
    assert len(params) == len(descriptions) >= 3
    assert any("plus" in d for d in descriptions)
    assert any(d.startswith("textencoder") for d in descriptions)
    lrs = sorted({g["lr"] for g in params})
    assert 0.5 in lrs and 1e-4 in lrs and 4e-3 in lrs

    keys_scaled, mean_norm, max_norm = network.apply_max_norm_regularization(1.0, torch.device("cpu"))
    assert keys_scaled == 0 and mean_norm == 0.0 and max_norm == 0.0
