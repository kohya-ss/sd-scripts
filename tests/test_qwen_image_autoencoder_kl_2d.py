import torch

from library.qwen_image_autoencoder_kl import AutoencoderKLQwenImage
from library.qwen_image_autoencoder_kl_2d import AutoencoderKLQwenImage2D, convert_3d_state_dict_to_2d


def test_qwen_image_2d_vae_matches_single_frame_3d_vae():
    kwargs = dict(
        base_dim=4,
        z_dim=2,
        dim_mult=[1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False],
        dropout=0.0,
        latents_mean=[0.0, 0.0],
        latents_std=[1.0, 1.0],
        input_channels=3,
    )

    torch.manual_seed(0)
    vae_3d = AutoencoderKLQwenImage(**kwargs).float().eval()
    vae_2d = AutoencoderKLQwenImage2D(**kwargs).float().eval()
    vae_2d.load_state_dict(convert_3d_state_dict_to_2d(vae_3d.state_dict()), strict=True)

    pixels = torch.randn(1, 3, 1, 8, 8)
    latents = torch.randn(1, 2, 1, 8, 8)

    with torch.no_grad():
        encoded_3d = vae_3d.encode(pixels, return_dict=False)[0].mode().squeeze(2)
        encoded_2d = vae_2d.encode(pixels.squeeze(2), return_dict=False)[0].mode()
        decoded_3d = vae_3d.decode(latents, return_dict=False)[0].squeeze(2)
        decoded_2d = vae_2d.decode(latents.squeeze(2), return_dict=False)[0]

    assert torch.allclose(encoded_2d, encoded_3d)
    assert torch.allclose(decoded_2d, decoded_3d, atol=1e-6)


def test_qwen_image_2d_state_dict_conversion_drops_temporal_weights():
    state_dict = {
        "encoder.conv_in.weight": torch.randn(4, 3, 3, 3, 3),
        "encoder.conv_in.bias": torch.randn(4),
        "encoder.norm_out.gamma": torch.randn(4, 1, 1, 1),
        "encoder.down_blocks.0.time_conv.weight": torch.randn(4, 4, 3, 1, 1),
    }

    converted = convert_3d_state_dict_to_2d(state_dict)

    assert converted["encoder.conv_in.weight"].shape == (4, 3, 3, 3)
    assert torch.equal(converted["encoder.conv_in.weight"], state_dict["encoder.conv_in.weight"][:, :, -1])
    assert converted["encoder.norm_out.gamma"].shape == (4, 1, 1)
    assert "encoder.down_blocks.0.time_conv.weight" not in converted
