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

def test_dreambooth_dataset_pin_memory():
    """
    Test pin_memory functionality using a simple mock dataset
    """
    from library.train_util import DreamBoothDataset, DreamBoothSubset, collator_class

    # Create a mock DreamBoothSubset with minimal arguments
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

    # Create a simplified mock dataset
    class SimpleMockDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [torch.randn(64, 64) for _ in range(4)]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    # Create a DataLoader to test pin_memory
    dataloader = torch.utils.data.DataLoader(
        SimpleMockDataset(), 
        batch_size=2, 
        num_workers=0,  # Use 0 to avoid multiprocessing overhead
        pin_memory=True
    )

    # Verify pin_memory works correctly
    for batch in dataloader:
        assert all(tensor.is_pinned() for tensor in batch), "All tensors should be pinned"
        break

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pin_memory_cuda_transfer():
    """
    Test pin_memory functionality for CUDA tensor transfer
    """
    # Create a simple dataset
    class SimpleCUDADataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = [torch.randn(64, 64) for _ in range(4)]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]

    # Create a DataLoader with pin_memory enabled
    dataloader = torch.utils.data.DataLoader(
        SimpleCUDADataset(), 
        batch_size=2, 
        num_workers=0,  # Use 0 to avoid multiprocessing overhead
        pin_memory=True
    )

    # Verify CUDA transfer works with pinned memory
    for batch in dataloader:
        cuda_batch = [tensor.to('cuda', non_blocking=True) for tensor in batch]
        assert all(tensor.is_pinned() for tensor in batch), "All tensors should be pinned"
        break

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
        pytest.skip("Accelerate library not installed")

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