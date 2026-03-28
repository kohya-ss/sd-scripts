from pathlib import Path

import torch

from library.leco_train_util import load_prompt_settings


def test_load_prompt_settings_with_original_format(tmp_path: Path):
    prompt_file = tmp_path / "prompts.toml"
    prompt_file.write_text(
        """
[[prompts]]
target = "van gogh"
guidance_scale = 1.5
resolution = 512
""".strip(),
        encoding="utf-8",
    )

    prompts = load_prompt_settings(prompt_file)

    assert len(prompts) == 1
    assert prompts[0].target == "van gogh"
    assert prompts[0].positive == "van gogh"
    assert prompts[0].unconditional == ""
    assert prompts[0].neutral == ""
    assert prompts[0].action == "erase"
    assert prompts[0].guidance_scale == 1.5


def test_load_prompt_settings_with_slider_targets(tmp_path: Path):
    prompt_file = tmp_path / "slider.toml"
    prompt_file.write_text(
        """
guidance_scale = 2.0
resolution = 768
neutral = ""

[[targets]]
target_class = ""
positive = "high detail"
negative = "low detail"
multiplier = 1.25
weight = 0.5
""".strip(),
        encoding="utf-8",
    )

    prompts = load_prompt_settings(prompt_file)

    assert len(prompts) == 4

    first = prompts[0]
    second = prompts[1]
    third = prompts[2]
    fourth = prompts[3]

    assert first.target == ""
    assert first.positive == "low detail"
    assert first.unconditional == "high detail"
    assert first.action == "erase"
    assert first.multiplier == 1.25
    assert first.weight == 0.5
    assert first.get_resolution() == (768, 768)

    assert second.positive == "high detail"
    assert second.unconditional == "low detail"
    assert second.action == "enhance"
    assert second.multiplier == 1.25

    assert third.action == "erase"
    assert third.multiplier == -1.25

    assert fourth.action == "enhance"
    assert fourth.multiplier == -1.25


def test_predict_noise_xl_uses_vector_embedding_from_add_time_ids():
    from library import sdxl_train_util
    from library.leco_train_util import PromptEmbedsXL, predict_noise_xl

    class DummyScheduler:
        def scale_model_input(self, latent_model_input, timestep):
            return latent_model_input

    class DummyUNet:
        def __call__(self, x, timesteps, context, y):
            self.x = x
            self.timesteps = timesteps
            self.context = context
            self.y = y
            return torch.zeros_like(x)

    latents = torch.randn(1, 4, 8, 8)
    prompt_embeds = PromptEmbedsXL(
        text_embeds=torch.randn(2, 77, 2048),
        pooled_embeds=torch.randn(2, 1280),
    )
    add_time_ids = torch.tensor(
        [
            [1024, 1024, 0, 0, 1024, 1024],
            [1024, 1024, 0, 0, 1024, 1024],
        ],
        dtype=prompt_embeds.pooled_embeds.dtype,
    )

    unet = DummyUNet()
    noise_pred = predict_noise_xl(unet, DummyScheduler(), torch.tensor(10), latents, prompt_embeds, add_time_ids)

    expected_size_embeddings = sdxl_train_util.get_size_embeddings(
        add_time_ids[:, :2], add_time_ids[:, 2:4], add_time_ids[:, 4:6], latents.device
    ).to(prompt_embeds.pooled_embeds.dtype)

    assert noise_pred.shape == latents.shape
    assert unet.context is prompt_embeds.text_embeds
    assert torch.equal(unet.y, torch.cat([prompt_embeds.pooled_embeds, expected_size_embeddings], dim=1))
