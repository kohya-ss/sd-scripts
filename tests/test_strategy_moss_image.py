import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library import moss_image_utils, strategy_moss_image
from library.strategy_moss_image import (
    MossImageLatentsCachingStrategy,
    MossImageTextEncoderOutputsCachingStrategy,
    MossImageTextEncodingStrategy,
    MossImageTokenizeStrategy,
    bf16_to_uint16,
    uint16_to_float32,
)


class FakeTokenizer:
    """Character level tokenizer with the interface used by moss_image_utils.tokenize_prompts."""

    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
        ids = [ord(c) % 250 + 1 for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.int64)}


class FakeTextEncoder(torch.nn.Module):
    """Deterministic stand-in for Qwen3_5TextModel: output depends on the token and its position."""

    def __init__(self, hidden_size=8):
        super().__init__()
        self.embed = torch.nn.Embedding(256, hidden_size)
        torch.manual_seed(0)
        self.embed.weight.data.normal_()

    def forward(self, input_ids, use_cache=False, return_dict=True):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :, None].float()
        return SimpleNamespace(last_hidden_state=(self.embed(input_ids) + pos * 0.01).to(torch.bfloat16))


def make_tokenize_strategy(max_length=4096):
    strategy = MossImageTokenizeStrategy.__new__(MossImageTokenizeStrategy)
    strategy.tokenizer = FakeTokenizer()
    strategy.max_length = max_length
    strategy.pad_token_id = 0
    return strategy


def test_bf16_bits_roundtrip():
    t = torch.randn(5, 7).to(torch.bfloat16)
    bits = bf16_to_uint16(t)
    assert bits.dtype == np.uint16 and bits.shape == (5, 7)
    restored = uint16_to_float32(bits)
    assert restored.dtype == np.float32
    assert torch.equal(torch.from_numpy(restored), t.float())


def test_tokenize_pads_and_masks():
    strategy = make_tokenize_strategy()
    input_ids, mask = strategy.tokenize(["ab", "abcd"])
    template_len = len(moss_image_utils.format_prompt(""))
    assert input_ids.shape == (2, template_len + 4) and mask.shape == input_ids.shape
    assert mask[0].sum() == template_len + 2 and mask[1].sum() == template_len + 4
    assert torch.all(input_ids[0, template_len + 2 :] == 0)
    # single string input gives a batch of one
    single_ids, single_mask = strategy.tokenize("ab")
    assert single_ids.shape[0] == 1 and torch.equal(single_ids[0], input_ids[0, : single_ids.shape[1]])


def test_tokenize_truncates():
    strategy = make_tokenize_strategy(max_length=5)
    input_ids, mask = strategy.tokenize("a long prompt")
    assert input_ids.shape == (1, 5) and mask.sum() == 5


def test_encode_tokens_matches_per_sample_encoding():
    tokenize_strategy = make_tokenize_strategy()
    encoding_strategy = MossImageTextEncodingStrategy()
    text_encoder = FakeTextEncoder()
    captions = ["short", "a longer caption"]
    tokens = tokenize_strategy.tokenize(captions)
    embeds, mask = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
    assert embeds.dtype == torch.bfloat16 and embeds.shape[0] == 2 and mask.dtype == torch.int32
    lens = tokens[1].sum(1).tolist()
    assert mask.sum(1).tolist() == lens
    for i, caption in enumerate(captions):
        ids = moss_image_utils.tokenize_prompts(tokenize_strategy.tokenizer, [caption])[0]
        expected = text_encoder(ids[None]).last_hidden_state[0]
        assert torch.equal(embeds[i, : lens[i]], expected)
        assert torch.all(embeds[i, lens[i] :] == 0)


def test_drop_cached_text_encoder_outputs():
    encoding_strategy = MossImageTextEncodingStrategy()
    prompt_embeds = torch.randn(2, 6, 4)
    attn_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.int32)
    uncond_embeds = torch.randn(2, 9, 4)  # the unconditional prompt can be longer than the batch
    uncond_mask = torch.ones(2, 9, dtype=torch.int32)

    # no dropout: unchanged
    out = encoding_strategy.drop_cached_text_encoder_outputs(prompt_embeds, attn_mask, uncond_embeds, uncond_mask, torch.zeros(2))
    assert out[0] is prompt_embeds and out[1] is attn_mask

    # drop everything: replaced by the unconditional embeddings and re-padded to its length
    out_embeds, out_mask = encoding_strategy.drop_cached_text_encoder_outputs(
        prompt_embeds, attn_mask, uncond_embeds, uncond_mask, torch.ones(2)
    )
    assert out_embeds.shape == (2, 9, 4) and torch.equal(out_embeds, uncond_embeds) and torch.all(out_mask == 1)

    # drop only the second sample
    out_embeds, out_mask = encoding_strategy.drop_cached_text_encoder_outputs(
        prompt_embeds, attn_mask, uncond_embeds, uncond_mask, torch.tensor([0.0, 1.0])
    )
    assert out_embeds.shape == (2, 9, 4)
    assert torch.equal(out_embeds[0, :3], prompt_embeds[0, :3]) and torch.all(out_embeds[0, 3:] == 0)
    assert out_mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert torch.equal(out_embeds[1], uncond_embeds[1]) and torch.all(out_mask[1] == 1)


@pytest.mark.parametrize("cache_to_disk", [True, False])
def test_text_encoder_outputs_caching_roundtrip(tmp_path, cache_to_disk):
    tokenize_strategy = make_tokenize_strategy()
    encoding_strategy = MossImageTextEncodingStrategy()
    text_encoder = FakeTextEncoder()
    caching_strategy = MossImageTextEncoderOutputsCachingStrategy(cache_to_disk, 2, False)

    captions = ["a cat", "a very long caption about a dog"]
    infos = []
    for i, caption in enumerate(captions):
        image_path = str(tmp_path / f"img{i}.png")
        info = SimpleNamespace(
            absolute_path=image_path,
            caption=caption,
            caption_dropout_rate=0.25 * i,
            text_encoder_outputs_npz=caching_strategy.get_outputs_npz_path(image_path),
            text_encoder_outputs=None,
        )
        infos.append(info)
    assert infos[0].text_encoder_outputs_npz.endswith("_moss_te.npz")
    assert not caching_strategy.is_disk_cached_outputs_expected(infos[0].text_encoder_outputs_npz)

    caching_strategy.cache_batch_outputs(tokenize_strategy, [text_encoder], encoding_strategy, infos, captions)

    # expected values: per-sample encoding (bf16) and the unconditional embeddings
    tokens = tokenize_strategy.tokenize(captions)
    expected_embeds, expected_mask = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
    uncond_tokens = tokenize_strategy.tokenize("")
    expected_uncond, _ = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], uncond_tokens)

    for i, info in enumerate(infos):
        if cache_to_disk:
            assert os.path.exists(info.text_encoder_outputs_npz)
            assert caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
            outputs = caching_strategy.load_outputs_npz(info.text_encoder_outputs_npz)
            assert info.text_encoder_outputs is None
        else:
            # in memory: one tuple per caption candidate
            assert isinstance(info.text_encoder_outputs, list) and len(info.text_encoder_outputs) == 1
            outputs = info.text_encoder_outputs[0]
        prompt_embeds, attn_mask, uncond_embeds, uncond_mask, rate = outputs
        length = int(expected_mask[i].sum())
        assert prompt_embeds.dtype == np.float32 and prompt_embeds.shape == (length, 8)  # trimmed, no padding
        assert torch.equal(torch.from_numpy(prompt_embeds), expected_embeds[i, :length].float())
        assert attn_mask.dtype == np.int32 and attn_mask.shape == (length,) and attn_mask.all()
        assert torch.equal(torch.from_numpy(uncond_embeds), expected_uncond[0].float())
        assert uncond_mask.shape == (expected_uncond.shape[1],) and uncond_mask.all()
        assert float(rate) == pytest.approx(0.25 * i)
        # the batch converter of the dataset (torch.FloatTensor) must accept these arrays
        assert torch.FloatTensor(prompt_embeds).shape == prompt_embeds.shape
        assert torch.FloatTensor(attn_mask).tolist() == [1.0] * length

    # the unconditional embeddings are encoded once and reused
    assert caching_strategy._uncond_embeds is not None
    first = caching_strategy._uncond_embeds
    caching_strategy.cache_batch_outputs(tokenize_strategy, [text_encoder], encoding_strategy, infos[:1], captions[:1])
    assert caching_strategy._uncond_embeds is first


def make_info(tmp_path, caching_strategy, name, caption, rate=0.0):
    image_path = str(tmp_path / f"{name}.png")
    return SimpleNamespace(
        absolute_path=image_path,
        caption=caption,
        caption_dropout_rate=rate,
        text_encoder_outputs_npz=caching_strategy.get_outputs_npz_path(image_path),
        text_encoder_outputs=None,
    )


@pytest.mark.parametrize("cache_to_disk", [True, False])
def test_text_encoder_outputs_caching_multiple_candidates(tmp_path, cache_to_disk):
    tokenize_strategy = make_tokenize_strategy()
    encoding_strategy = MossImageTextEncodingStrategy()
    text_encoder = FakeTextEncoder()
    caching_strategy = MossImageTextEncoderOutputsCachingStrategy(cache_to_disk, 4, False)
    assert caching_strategy.supports_multiple_caption_candidates

    captions_list = [["a cat", "a photo of\na cat", "cat, animal"], ["a dog"]]
    infos = [make_info(tmp_path, caching_strategy, f"img{i}", "raw", rate=0.1 * i) for i in range(len(captions_list))]

    caching_strategy.cache_batch_outputs_multi(tokenize_strategy, [text_encoder], encoding_strategy, infos, captions_list)

    flat_captions = [c for captions in captions_list for c in captions]
    tokens = tokenize_strategy.tokenize(flat_captions)
    expected_embeds, expected_mask = encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)

    offset = 0
    for i, (info, captions) in enumerate(zip(infos, captions_list)):
        if cache_to_disk:
            npz = np.load(info.text_encoder_outputs_npz)
            assert int(npz["num_candidates"]) == len(captions)
            assert [str(npz[f"caption_{k}"]) for k in range(len(captions))] == captions
            assert "prompt_embeds" not in npz  # the new format always uses the _{k} keys
            # valid only for the same candidates
            assert caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, captions)
            assert caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)  # no caption check
            assert not caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, captions[:1] * 2)
            assert not caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, captions + ["x"])
            assert not caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, captions[:-1] + ["x"])
        else:
            assert len(info.text_encoder_outputs) == len(captions)

        for k in range(len(captions)):
            if cache_to_disk:
                outputs = caching_strategy.load_outputs_npz(info.text_encoder_outputs_npz, k)
            else:
                outputs = info.text_encoder_outputs[k]
            prompt_embeds, attn_mask, uncond_embeds, uncond_mask, rate = outputs
            length = int(expected_mask[offset + k].sum())
            assert prompt_embeds.shape == (length, 8)
            assert torch.equal(torch.from_numpy(prompt_embeds), expected_embeds[offset + k, :length].float())
            assert attn_mask.shape == (length,) and attn_mask.all()
            assert float(rate) == pytest.approx(0.1 * i)
        offset += len(captions)

        if cache_to_disk:
            with pytest.raises(ValueError, match="skip_cache_check"):
                caching_strategy.load_outputs_npz(info.text_encoder_outputs_npz, len(captions))


def test_text_encoder_outputs_caching_skip_check_ignores_captions(tmp_path):
    tokenize_strategy = make_tokenize_strategy()
    encoding_strategy = MossImageTextEncodingStrategy()
    text_encoder = FakeTextEncoder()
    caching_strategy = MossImageTextEncoderOutputsCachingStrategy(True, 4, skip_disk_cache_validity_check=True)
    info = make_info(tmp_path, caching_strategy, "img", "raw")
    assert not caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, ["a"])  # no file yet
    caching_strategy.cache_batch_outputs_multi(tokenize_strategy, [text_encoder], encoding_strategy, [info], [["a"]])
    assert caching_strategy.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz, ["changed", "captions"])


def test_text_encoder_outputs_previous_format_is_accepted_for_single_candidate(tmp_path):
    caching_strategy = MossImageTextEncoderOutputsCachingStrategy(True, 4, False)
    npz_path = str(tmp_path / "img_moss_te.npz")
    embeds = bf16_to_uint16(torch.randn(3, 8).to(torch.bfloat16))
    uncond = bf16_to_uint16(torch.randn(2, 8).to(torch.bfloat16))
    np.savez(
        npz_path,
        prompt_embeds=embeds,
        attn_mask=np.ones(3, dtype=np.int32),
        uncond_embeds=uncond,
        uncond_mask=np.ones(2, dtype=np.int32),
        caption_dropout_rate=np.array(0.0, dtype=np.float32),
    )

    assert caching_strategy.is_disk_cached_outputs_expected(npz_path)
    assert caching_strategy.is_disk_cached_outputs_expected(npz_path, ["any text"])  # the text cannot be checked
    assert not caching_strategy.is_disk_cached_outputs_expected(npz_path, ["a", "b"])  # must be re-cached

    outputs = caching_strategy.load_outputs_npz(npz_path)
    assert np.array_equal(outputs[0], uint16_to_float32(embeds))
    assert np.array_equal(outputs[2], uint16_to_float32(uncond))
    with pytest.raises(ValueError, match="1 caption candidate"):
        caching_strategy.load_outputs_npz(npz_path, 1)


class FakeVAE(torch.nn.Module):
    """Returns 32 channel latents at 1/8 resolution: average pooling of the input, repeated over channels."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.dummy.device

    @property
    def dtype(self):
        return self.dummy.dtype

    def encode(self, x):
        pooled = torch.nn.functional.avg_pool2d(x, 8)  # [B, 3, H/8, W/8]
        return pooled.repeat(1, 11, 1, 1)[:, :32]


def test_latents_caching_roundtrip(tmp_path):
    from PIL import Image

    caching_strategy = MossImageLatentsCachingStrategy(True, 2, False)
    assert caching_strategy.cache_suffix == "_moss.npz"

    infos = []
    for i, (w, h) in enumerate([(64, 48), (64, 48)]):
        image_path = str(tmp_path / f"img{i}.png")
        rng = np.random.RandomState(i)
        Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(image_path)
        npz_path = caching_strategy.get_latents_npz_path(image_path, (w, h))
        assert npz_path.endswith("_0064x0048_moss.npz")
        info = SimpleNamespace(
            absolute_path=image_path,
            image_key=image_path,
            image=None,
            bucket_reso=(w, h),
            resized_size=(w, h),
            latents_npz=npz_path,
            latents=None,
            latents_flipped=None,
            latents_original_size=None,
            latents_crop_ltrb=None,
            alpha_mask=None,
            resize_interpolation=None,
        )
        infos.append(info)
    bucket_reso = (64, 48)
    assert not caching_strategy.is_disk_cached_latents_expected(bucket_reso, infos[0].latents_npz, flip_aug=False, alpha_mask=False)

    vae = FakeVAE()
    caching_strategy.cache_batch_latents(vae, infos, flip_aug=True, alpha_mask=False, random_crop=False)

    for info in infos:
        assert caching_strategy.is_disk_cached_latents_expected(bucket_reso, info.latents_npz, flip_aug=True, alpha_mask=False)
        latents, original_size, crop_ltrb, flipped, alpha = caching_strategy.load_latents_from_disk(info.latents_npz, bucket_reso)
        assert latents.shape == (32, 6, 8) and flipped.shape == (32, 6, 8) and alpha is None
        assert original_size == [64, 48]
        # the stored values are the raw VAE outputs (no patchify / normalization)
        image = Image.open(info.absolute_path)
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        expected = vae.encode(pixels)[0]
        assert torch.allclose(torch.from_numpy(latents), expected, atol=1e-5)
        assert torch.allclose(torch.from_numpy(flipped), vae.encode(torch.flip(pixels, dims=[3]))[0], atol=1e-5)
