import os
import sys
import unittest
import math
import random
import types
import importlib.machinery

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Avoid importing broken xformers CUDA extensions while importing diffusers in lightweight dataset tests.
if "xformers" not in sys.modules:
    xformers_module = types.ModuleType("xformers")
    xformers_ops_module = types.ModuleType("xformers.ops")
    xformers_module.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None)
    xformers_ops_module.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = xformers_ops_module
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = xformers_ops_module

if "diffusers" not in sys.modules:
    diffusers_module = types.ModuleType("diffusers")
    diffusers_module.__spec__ = importlib.machinery.ModuleSpec("diffusers", loader=None)

    class _DummyDiffusersClass:
        pass

    diffusers_module.AutoencoderKL = _DummyDiffusersClass
    diffusers_module.DDIMScheduler = _DummyDiffusersClass
    diffusers_module.EulerAncestralDiscreteScheduler = _DummyDiffusersClass
    diffusers_module.StableDiffusionPipeline = _DummyDiffusersClass
    sys.modules["diffusers"] = diffusers_module
    diffusers_schedulers_module = types.ModuleType("diffusers.schedulers")
    diffusers_schedulers_module.__spec__ = importlib.machinery.ModuleSpec("diffusers.schedulers", loader=None)
    diffusers_euler_module = types.ModuleType("diffusers.schedulers.scheduling_euler_ancestral_discrete")
    diffusers_euler_module.__spec__ = importlib.machinery.ModuleSpec(
        "diffusers.schedulers.scheduling_euler_ancestral_discrete", loader=None
    )
    diffusers_euler_module.EulerAncestralDiscreteSchedulerOutput = _DummyDiffusersClass
    sys.modules["diffusers.schedulers"] = diffusers_schedulers_module
    sys.modules["diffusers.schedulers.scheduling_euler_ancestral_discrete"] = diffusers_euler_module

from library.subset import BaseSubset
from library.dataset import BaseDataset, ImageInfo
from library.dreambooth_dataset import DreamBoothDataset
from library.finetuning_dataset import FineTuningDataset
from library.controlnet_dataset import ControlNetDataset

class TestFADImplementation(unittest.TestCase):
    def setUp(self):
        random.seed(42)

    def test_fad_frequencies_no_wildcard(self):
        # 1. Setup mock subset
        subset = BaseSubset(
            image_dir="/mock/img/dir",
            alpha_mask=False,
            num_repeats=1,
            shuffle_caption=False,
            caption_separator=",",
            keep_tokens=0,
            keep_tokens_separator=None,
            secondary_separator=None,
            enable_wildcard=False,
            color_aug=False,
            flip_aug=False,
            face_crop_aug_range=None,
            random_crop=False,
            caption_dropout_rate=0.0,
            caption_dropout_every_n_epochs=0,
            caption_tag_dropout_rate=0.0,
            caption_prefix=None,
            caption_suffix=None,
            token_warmup_min=0,
            token_warmup_step=0,
            enable_fad=True,
            fad_curriculum=False,
        )

        # 2. Setup BaseDataset
        dataset = BaseDataset(
            resolution=(512, 512),
            network_multiplier=1.0,
            train_inpainting=False,
            debug_dataset=False,
            enable_fad=True,
            fad_curriculum=False,
        )
        dataset.batch_size = 1
        dataset.subsets = [subset]

        # 3. Register mock images and captions
        # Total 3 images
        # Caption 1: "trigger, cat, animal"
        # Caption 2: "trigger, dog, animal"
        # Caption 3: "trigger, bird"
        # Since keep_tokens = 0, all tokens are flex_tokens
        captions = [
            "trigger, cat, animal",
            "trigger, dog, animal",
            "trigger, bird",
        ]
        
        for idx, caption in enumerate(captions):
            info = ImageInfo(
                image_key=f"img_{idx}",
                num_repeats=1,
                caption=caption,
                is_reg=False,
                absolute_path=f"/mock/img/dir/img_{idx}.jpg",
            )
            # mock get_image_size to avoid calling PIL
            info.image_size = (512, 512)
            dataset.register_image(info, subset)

        # 4. Run make_buckets to initialize FAD statistics
        dataset.make_buckets()

        # 5. Check computed frequencies r(w)
        # N_t = 3
        # "trigger" appears in all 3 -> 3/3 = 1.0
        # "animal" appears in 2 -> 2/3 = 0.6666...
        # "cat", "dog", "bird" each appear in 1 -> 1/3 = 0.3333...
        freq = subset.fad_tag_frequencies
        
        self.assertAlmostEqual(freq.get("trigger", 0.0), 1.0)
        self.assertAlmostEqual(freq.get("animal", 0.0), 2.0 / 3.0)
        self.assertAlmostEqual(freq.get("cat", 0.0), 1.0 / 3.0)
        self.assertAlmostEqual(freq.get("dog", 0.0), 1.0 / 3.0)
        self.assertAlmostEqual(freq.get("bird", 0.0), 1.0 / 3.0)

    def test_fad_frequencies_with_wildcard_uses_exact_expectation(self):
        # 1. Setup mock subset with enable_wildcard=True
        subset = BaseSubset(
            image_dir="/mock/img/dir",
            alpha_mask=False,
            num_repeats=1,
            shuffle_caption=False,
            caption_separator=",",
            keep_tokens=0,
            keep_tokens_separator=None,
            secondary_separator=None,
            enable_wildcard=True,
            color_aug=False,
            flip_aug=False,
            face_crop_aug_range=None,
            random_crop=False,
            caption_dropout_rate=0.0,
            caption_dropout_every_n_epochs=0,
            caption_tag_dropout_rate=0.0,
            caption_prefix=None,
            caption_suffix=None,
            token_warmup_min=0,
            token_warmup_step=0,
            enable_fad=True,
            fad_curriculum=True,
        )

        dataset = BaseDataset(
            resolution=(512, 512),
            network_multiplier=1.0,
            train_inpainting=False,
            debug_dataset=False,
            enable_fad=True,
            fad_curriculum=True,
        )
        dataset.batch_size = 1
        dataset.subsets = [subset]

        # Register 1 image with multiline and {choice} wildcard
        # Line 1: "trigger, cat"
        # Line 2: "trigger, {dog|bird}"
        # Expected occurrences:
        # Line 1: 0.5 probability
        #   "cat" -> 0.5 * 1.0 = 0.5 expected count
        # Line 2: 0.5 probability
        #   "dog" -> 0.5 * 0.5 = 0.25 expected count
        #   "bird" -> 0.5 * 0.5 = 0.25 expected count
        #   "trigger" -> 0.5 * 1.0 (from line 1) + 0.5 * 1.0 (from line 2) = 1.0 expected count
        caption = "trigger, cat\ntrigger, {dog|bird}"
        info = ImageInfo(
            image_key="img_wildcard",
            num_repeats=1,
            caption=caption,
            is_reg=False,
            absolute_path="/mock/img/dir/img_wildcard.jpg",
        )
        info.image_size = (512, 512)
        dataset.register_image(info, subset)

        dataset.make_buckets()

        freq = subset.fad_tag_frequencies
        
        self.assertAlmostEqual(freq.get("trigger", 0.0), 1.0)
        self.assertAlmostEqual(freq.get("cat", 0.0), 0.5)
        self.assertAlmostEqual(freq.get("dog", 0.0), 0.25)
        self.assertAlmostEqual(freq.get("bird", 0.0), 0.25)

    def test_fad_curriculum_steps(self):
        # Setup FAD with curriculum enabled
        subset = BaseSubset(
            image_dir="/mock/img/dir",
            alpha_mask=False,
            num_repeats=1,
            shuffle_caption=False,
            caption_separator=",",
            keep_tokens=0,
            keep_tokens_separator=None,
            secondary_separator=None,
            enable_wildcard=False,
            color_aug=False,
            flip_aug=False,
            face_crop_aug_range=None,
            random_crop=False,
            caption_dropout_rate=0.0,
            caption_dropout_every_n_epochs=0,
            caption_tag_dropout_rate=0.0,
            caption_prefix=None,
            caption_suffix=None,
            token_warmup_min=0,
            token_warmup_step=0,
            enable_fad=True,
            fad_curriculum=True,
        )

        dataset = BaseDataset(
            resolution=(512, 512),
            network_multiplier=1.0,
            train_inpainting=False,
            debug_dataset=False,
            enable_fad=True,
            fad_curriculum=True,
            fad_curriculum_start=0.1,
            fad_curriculum_end=0.8,
            fad_curriculum_beta=3.0,
            fad_step_start=0.0,
            fad_step_end=1.0,
            fad_p_min=0.35,
            fad_p_max=1.0,
            fad_alpha=10.0,
            fad_c=0.5,
        )
        dataset.batch_size = 1
        dataset.subsets = [subset]

        # Register 1 image
        info = ImageInfo(
            image_key="img",
            num_repeats=1,
            caption="trigger, cat",
            is_reg=False,
            absolute_path="/mock/img/dir/img.jpg",
        )
        info.image_size = (512, 512)
        dataset.register_image(info, subset)
        dataset.make_buckets()

        # Set max training steps to 1000
        dataset.set_max_train_steps(1000)

        # 1. At step 50 (below warmup 0.1 * 1000 = 100)
        # p_step should be fad_step_start = 0.0 by Eq. (7)
        dataset.set_current_step(50)
        
        # We'll run process_caption multiple times to observe dropout rate for "cat"
        # "cat" frequency is 1.0. 
        # p_drop = 0.35 + 0.65 * sigmoid(10 * (1.0 - 0.5)) = 0.35 + 0.65 * sigmoid(5) = 0.35 + 0.65 * 0.9933 = 0.9956
        # p_drop_final = p_drop * p_step = 0.9956 * 0.0 = 0
        # Expectation of cat survival: 100%
        survived = 0
        for _ in range(500):
            res = dataset.process_caption(subset, "trigger, cat")
            if "cat" in res:
                survived += 1
        
        survival_rate = survived / 500
        self.assertEqual(survival_rate, 1.0, f"Survival rate: {survival_rate}")

        # 2. At step 900 (above end 0.8 * 1000 = 800)
        # p_step should be fad_step_end = 1.0 by Eq. (7)
        # p_drop_final = 0.9956 * 1.0 = 0.9956
        # Expectation of cat survival: near 0%
        dataset.set_current_step(900)
        survived = 0
        for _ in range(500):
            res = dataset.process_caption(subset, "trigger, cat")
            if "cat" in res:
                survived += 1
        
        survival_rate = survived / 500
        self.assertTrue(0.0 <= survival_rate < 0.03, f"Survival rate: {survival_rate}")

    def test_fad_makes_text_encoder_output_uncacheable(self):
        subset = BaseSubset(
            image_dir="/mock/img/dir",
            alpha_mask=False,
            num_repeats=1,
            shuffle_caption=False,
            caption_separator=",",
            keep_tokens=0,
            keep_tokens_separator=None,
            secondary_separator=None,
            enable_wildcard=False,
            color_aug=False,
            flip_aug=False,
            face_crop_aug_range=None,
            random_crop=False,
            caption_dropout_rate=0.0,
            caption_dropout_every_n_epochs=0,
            caption_tag_dropout_rate=0.0,
            caption_prefix=None,
            caption_suffix=None,
            token_warmup_min=0,
            token_warmup_step=0,
            enable_fad=True,
            fad_curriculum=False,
        )
        dataset = BaseDataset(
            resolution=(512, 512),
            network_multiplier=1.0,
            train_inpainting=False,
            debug_dataset=False,
            enable_fad=True,
            fad_curriculum=False,
        )
        dataset.subsets = [subset]

        self.assertFalse(dataset.is_text_encoder_output_cacheable(cache_supports_dropout=True))

    def test_dataset_classes_accept_dataset_level_fad_params(self):
        common_params = dict(
            subsets=[],
            batch_size=1,
            resolution=(512, 512),
            network_multiplier=1.0,
            enable_bucket=False,
            min_bucket_reso=256,
            max_bucket_reso=1024,
            bucket_reso_steps=64,
            bucket_no_upscale=False,
            train_inpainting=False,
            debug_dataset=False,
            validation_split=0.0,
            validation_seed=None,
            resize_interpolation=None,
            skip_image_resolution=None,
            enable_fad=True,
            fad_curriculum=True,
            fad_p_min=0.2,
            fad_p_max=0.9,
            fad_alpha=8.0,
            fad_c=0.4,
            fad_curriculum_start=0.2,
            fad_curriculum_end=0.7,
            fad_curriculum_beta=2.0,
            fad_step_start=0.0,
            fad_step_end=1.0,
        )

        dreambooth_dataset = DreamBoothDataset(is_training_dataset=True, prior_loss_weight=1.0, **common_params)
        finetuning_dataset = FineTuningDataset(**common_params)
        controlnet_dataset = ControlNetDataset(**common_params)

        self.assertTrue(dreambooth_dataset.enable_fad)
        self.assertTrue(finetuning_dataset.enable_fad)
        self.assertTrue(controlnet_dataset.enable_fad)
        self.assertEqual(dreambooth_dataset.fad_p_min, 0.2)

if __name__ == "__main__":
    unittest.main()
