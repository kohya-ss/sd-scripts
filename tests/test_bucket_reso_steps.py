"""bucket_reso_steps must follow each architecture's own minimum.

NetworkTrainer (SD1/SD2) requires a multiple of 64. SDXL, FLUX, Lumina and
HunyuanImage override that with a smaller step. Calling the parent check first
rejects those smaller values before the override runs (issue #1915).
"""

from types import SimpleNamespace

import pytest

from library.dataset import BaseDataset
import train_network
from flux_train_network import FluxNetworkTrainer
from hunyuan_image_train_network import HunyuanImageNetworkTrainer
from lumina_train_network import LuminaNetworkTrainer


class _BucketDataset:
    def __init__(self, steps: int):
        self.bucket_reso_steps = steps

    verify_bucket_reso_steps = BaseDataset.verify_bucket_reso_steps


class _BucketGroup:
    def __init__(self, steps: int):
        self.datasets = [_BucketDataset(steps)]

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)


def _assert_steps(check, allowed, rejected):
    for steps in allowed:
        check(_BucketGroup(steps), _BucketGroup(steps))
    for steps in rejected:
        with pytest.raises(AssertionError, match="bucket_reso_steps"):
            check(_BucketGroup(steps), None)


def test_sd1_still_requires_bucket_reso_steps_divisible_by_64():
    trainer = train_network.NetworkTrainer()

    def check(train_group, val_group):
        trainer.assert_extra_args(SimpleNamespace(), train_group, val_group)

    _assert_steps(check, allowed=(64, 128), rejected=(32, 16))


def test_flux_lora_allows_bucket_reso_steps_of_32():
    trainer = FluxNetworkTrainer()
    args = SimpleNamespace(
        timestep_sampling="ddpm",
        discrete_flow_shift=1.0,
        model_type="flux",
        fp8_base_unet=False,
        cache_text_encoder_outputs=False,
        cache_text_encoder_outputs_to_disk=False,
        network_train_unet_only=False,
        max_token_length=None,
        blocks_to_swap=None,
        cpu_offload_checkpointing=False,
        split_mode=False,
    )

    def check(train_group, val_group):
        trainer.assert_extra_args(args, train_group, val_group)

    _assert_steps(check, allowed=(32, 64), rejected=(16, 48))


def test_lumina_lora_allows_bucket_reso_steps_of_16():
    trainer = LuminaNetworkTrainer()
    args = SimpleNamespace(
        cache_text_encoder_outputs=False,
        cache_text_encoder_outputs_to_disk=False,
        network_train_unet_only=True,
    )

    def check(train_group, val_group):
        trainer.assert_extra_args(args, train_group, val_group)

    _assert_steps(check, allowed=(16, 32), rejected=(8, 24))


def test_hunyuan_image_lora_allows_bucket_reso_steps_of_32():
    trainer = HunyuanImageNetworkTrainer()
    args = SimpleNamespace(
        mixed_precision="bf16",
        fp8_base=False,
        fp8_base_unet=False,
        fp8_scaled=False,
        cache_text_encoder_outputs=False,
        cache_text_encoder_outputs_to_disk=False,
    )

    def check(train_group, val_group):
        trainer.assert_extra_args(args, train_group, val_group)

    _assert_steps(check, allowed=(32, 64), rejected=(16, 48))
