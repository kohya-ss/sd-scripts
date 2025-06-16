import argparse
import pytest
import torch
import importlib.metadata
import multiprocessing

def test_pin_memory_argument():
    """
    Test that the pin_memory argument is correctly added to argument parsers
    """
    from library.train_util import add_training_arguments

    parser = argparse.ArgumentParser()
    add_training_arguments(parser, support_dreambooth=True)

    # Parse an empty list of arguments to check the default
    args = parser.parse_args([])
    assert hasattr(args, "pin_memory"), "pin_memory argument should be present in argument parser"
    assert args.pin_memory is False, "pin_memory should default to False"

def test_image_info_pin_memory():
    """
    Test pin_memory method in ImageInfo class
    """
    from library.train_util import ImageInfo

    # Create an ImageInfo instance with mock tensors
    image_info = ImageInfo(
        image_key='test_key', 
        num_repeats=1, 
        caption='test caption', 
        is_reg=False, 
        absolute_path='/test/path'
    )

    # Add mock tensors that can track pinning
    class MockTensor:
        def __init__(self):
            self.pinned = False
        
        def pin_memory(self):
            self.pinned = True
            return self

    # Set mock tensors
    image_info.latents = MockTensor()
    image_info.text_encoder_outputs1 = MockTensor()
    image_info.text_encoder_outputs2 = MockTensor()
    image_info.text_encoder_pool2 = MockTensor()
    image_info.alpha_mask = MockTensor()

    # Call pin_memory
    pinned_image_info = image_info.pin_memory()

    # Verify all tensors are pinned
    assert pinned_image_info.latents.pinned, "Latents should be pinned"
    assert pinned_image_info.text_encoder_outputs1.pinned, "Text encoder outputs1 should be pinned"
    assert pinned_image_info.text_encoder_outputs2.pinned, "Text encoder outputs2 should be pinned"
    assert pinned_image_info.text_encoder_pool2.pinned, "Text encoder pool2 should be pinned"
    assert pinned_image_info.alpha_mask.pinned, "Alpha mask should be pinned"

def test_dreambooth_dataset_pin_memory():
    """
    Test pin_memory method in DreamBoothDataset
    """
    from library.train_util import DreamBoothDataset, DreamBoothSubset

    # Create a mock DreamBoothSubset with default arguments
    def create_mock_subset():
        return DreamBoothSubset(
            image_dir='/mock/path',
            is_reg=False,
            class_tokens='test_token',
            caption_extension='.txt',
            alpha_mask=False,
            num_repeats=1,
            shuffle_caption=False,
            caption_separator=',',
            keep_tokens=0,
            keep_tokens_separator='',
            secondary_separator='',
            enable_wildcard=False,
            color_aug=False,
            flip_aug=False,
            face_crop_aug_range=None,
            random_crop=False,
            caption_dropout_rate=0,
            caption_dropout_every_n_epochs=0,
            caption_tag_dropout_rate=0,
            caption_prefix='',
            caption_suffix='',
            token_warmup_min=1,
            token_warmup_step=0,
            cache_info=False
        )

    # Create a mock DreamBoothDataset
    class MockDreamBoothDataset(DreamBoothDataset):
        def __init__(self):
            # Prepare subset
            subsets = [create_mock_subset()]

            # Call parent constructor with minimal required arguments
            super().__init__(
                subsets=subsets,
                is_training_dataset=True,
                batch_size=1,
                resolution=(512, 512),
                network_multiplier=1.0,
                enable_bucket=False,
                min_bucket_reso=None,
                max_bucket_reso=None,
                bucket_reso_steps=None,
                bucket_no_upscale=False,
                prior_loss_weight=1.0,
                debug_dataset=False,
                validation_split=0.0,
                validation_seed=None,
                resize_interpolation=None
            )
            
            # Add mock image data for pin_memory testing
            self.image_data = {
                'mock_tensor1': self._create_mock_tensor(),
                'mock_tensor2': self._create_mock_tensor()
            }
        
        def _create_mock_tensor(self):
            class MockTensor:
                def __init__(self):
                    self.pinned = False
                
                def pin_memory(self):
                    self.pinned = True
                    return self
            return MockTensor()

    # Create dataset
    dataset = MockDreamBoothDataset()

    # Verify initial state
    for tensor in dataset.image_data.values():
        assert not hasattr(tensor, 'pinned') or not tensor.pinned, "Tensors should not be pinned initially"

    # Call pin_memory
    dataset.pin_memory()

    # Verify all tensors are pinned
    for tensor in dataset.image_data.values():
        assert tensor.pinned, "All tensors in image_data should be pinned"

def test_collator_pin_memory_method():
    """
    Test that collator correctly calls pin_memory on the dataset
    """
    from library.train_util import collator_class, DreamBoothDataset, DreamBoothSubset

    # Create a mock dataset that tracks pin_memory calls
    class MockPinMemoryDataset(DreamBoothDataset):
        def __init__(self):
            # Prepare subset
            def create_mock_subset():
                return DreamBoothSubset(
                    image_dir='/mock/path',
                    is_reg=False,
                    class_tokens='test_token',
                    caption_extension='.txt',
                    alpha_mask=False,
                    num_repeats=1,
                    shuffle_caption=False,
                    caption_separator=',',
                    keep_tokens=0,
                    keep_tokens_separator='',
                    secondary_separator='',
                    enable_wildcard=False,
                    color_aug=False,
                    flip_aug=False,
                    face_crop_aug_range=None,
                    random_crop=False,
                    caption_dropout_rate=0,
                    caption_dropout_every_n_epochs=0,
                    caption_tag_dropout_rate=0,
                    caption_prefix='',
                    caption_suffix='',
                    token_warmup_min=1,
                    token_warmup_step=0,
                    cache_info=False
                )

            # Prepare subset
            subsets = [create_mock_subset()]

            # Call parent constructor with minimal required arguments
            super().__init__(
                subsets=subsets,
                is_training_dataset=True,
                batch_size=1,
                resolution=(512, 512),
                network_multiplier=1.0,
                enable_bucket=False,
                min_bucket_reso=None,
                max_bucket_reso=None,
                bucket_reso_steps=None,
                bucket_no_upscale=False,
                prior_loss_weight=1.0,
                debug_dataset=False,
                validation_split=0.0,
                validation_seed=None,
                resize_interpolation=None
            )
            self.pin_memory_called = False

        def pin_memory(self):
            self.pin_memory_called = True
            return self

    # Create a multiprocessing manager for current epoch and step
    mp_manager = multiprocessing.Manager()
    current_epoch = mp_manager.Value('i', 0)
    current_step = mp_manager.Value('i', 0)

    # Create a dataset and collator
    dataset = MockPinMemoryDataset()
    collator = collator_class(current_epoch, current_step, dataset)

    # Call pin_memory on the collator
    collator.pin_memory()

    # Verify pin_memory was called on the dataset
    assert dataset.pin_memory_called, "Collator should call pin_memory on the dataset"

def test_training_scripts_pin_memory_support():
    """
    Verify that multiple training scripts support pin_memory argument
    """
    training_scripts = [
        "fine_tune.py",
        "flux_train.py",
        "sd3_train.py",
        "sdxl_train.py",
        "train_network.py",
        "train_textual_inversion.py",
        "sdxl_train_control_net.py",
        "flux_train_control_net.py",
    ]

    from library.train_util import add_training_arguments

    for script in training_scripts:
        parser = argparse.ArgumentParser()
        add_training_arguments(parser, support_dreambooth=True)

        # Parse arguments to check pin_memory
        args = parser.parse_args([])
        assert hasattr(args, "pin_memory"), f"{script} should have pin_memory argument"

def test_accelerator_pin_memory_config():
    """
    Test that the Accelerator is configured with pin_memory option
    Checks compatibility and configuration based on Accelerate library version
    """
    from library.train_util import prepare_accelerator

    # Check Accelerate library version
    try:
        accelerate_version = importlib.metadata.version("accelerate")
        print(f"Accelerate library version: {accelerate_version}")
    except importlib.metadata.PackageNotFoundError:
        pytest.fail("Accelerate library not installed")

    # Minimal args to pass initial checks
    args = argparse.Namespace(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with=None,
        kwargs_handlers=[],
        deepspeed_plugin=None,
        pin_memory=True,
        logging_dir=None,
        torch_compile=False,
        log_prefix=None,
        ddp_gradient_as_bucket_view=False,
        ddp_static_graph=False,
        ddp_timeout=None,
        wandb_api_key=None,
        dynamo_backend="NO",
        deepspeed=False,
    )

    # Prepare accelerator
    accelerator = prepare_accelerator(args)

    # Check for dataloader_config
    assert hasattr(accelerator, "dataloader_config"), "Accelerator should have dataloader_config when pin_memory is enabled"
    assert accelerator.dataloader_config.non_blocking is True, "Dataloader should be configured with pin_memory"
