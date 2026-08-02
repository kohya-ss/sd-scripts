import torch
from dataclasses import asdict

from library.config_util import ControlNetSubsetParams
from library.custom_train_functions import apply_masked_loss
from library.subset import ControlNetSubset


def test_controlnet_subset_accepts_alpha_mask():
    # mirrors the production construction path: subset_klass(**asdict(params))
    params = ControlNetSubsetParams(image_dir="img", conditioning_data_dir="cond", alpha_mask=True)
    subset = ControlNetSubset(**asdict(params))
    assert subset.alpha_mask is True

    params = ControlNetSubsetParams(image_dir="img", conditioning_data_dir="cond")
    subset = ControlNetSubset(**asdict(params))
    assert subset.alpha_mask is False


def test_apply_masked_loss_prefers_alpha_masks_over_conditioning_images():
    loss = torch.ones(2, 4, 8, 8)
    batch = {
        # R channel of -1 would zero the loss if (wrongly) used as the mask
        "conditioning_images": torch.full((2, 3, 64, 64), -1.0),
        "alpha_masks": torch.ones(2, 64, 64),
    }
    out = apply_masked_loss(loss.clone(), batch)
    assert torch.allclose(out, loss)


def test_apply_masked_loss_falls_back_to_conditioning_images():
    # masked-loss workflow: alpha_masks present but None -> conditioning images are the masks
    loss = torch.ones(2, 4, 8, 8)
    batch = {
        "conditioning_images": torch.full((2, 3, 64, 64), -1.0),  # -> weight 0
        "alpha_masks": None,
    }
    out = apply_masked_loss(loss.clone(), batch)
    assert torch.allclose(out, torch.zeros_like(loss))


def test_apply_masked_loss_normalize_keeps_per_sample_scale():
    loss = torch.ones(2, 4, 8, 8)
    # sample 0: uniform 0.25; sample 1: left half 0.25, right half 1.0
    masks = torch.full((2, 64, 64), 0.25)
    masks[1, :, 32:] = 1.0
    batch = {"alpha_masks": masks}

    out = apply_masked_loss(loss.clone(), batch, normalize=True)
    # per-sample mean returns to ~1.0 regardless of mask area
    assert torch.allclose(out.mean(dim=(1, 2, 3)), torch.ones(2), atol=1e-5)
    # relative weighting within a sample is preserved (4x between the halves)
    left = out[1, :, :, :4].mean()
    right = out[1, :, :, 4:].mean()
    assert torch.allclose(right / left, torch.tensor(4.0), atol=1e-4)


def test_apply_masked_loss_without_masks_is_identity():
    loss = torch.rand(2, 4, 8, 8)
    out = apply_masked_loss(loss.clone(), {})
    assert torch.allclose(out, loss)
