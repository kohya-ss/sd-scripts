import pytest
from pathlib import Path
import tempfile

from library.flux_utils import get_checkpoint_paths


def test_get_checkpoint_paths():
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Scenario 1: Single safetensors file in root directory
        single_file = temp_path / "model.safetensors"
        single_file.touch()
        paths = get_checkpoint_paths(str(single_file))
        assert len(paths) == 1
        assert paths[0] == single_file


def test_multiple_root_checkpoint_paths():
    """
    Multiple single safetensors files in root directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Scenario 2:
        file1 = temp_path / "model1.safetensors"
        file2 = temp_path / "model2.safetensors"
        file1.touch()
        file2.touch()
        paths = get_checkpoint_paths(temp_path)
        assert len(paths) == 2
        assert set(paths) == {file1, file2}


def test_multipart_sharded_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Scenario 3: Sharded multi-part checkpoint
        # Create sharded checkpoint files
        base_name = "diffusion_pytorch_model"
        total_parts = 3
        for i in range(1, total_parts + 1):
            (temp_path / f"{base_name}-{i:05d}-of-{total_parts:05d}.safetensors").touch()

        paths = get_checkpoint_paths(temp_path)
        assert len(paths) == total_parts

        # Check if all expected part paths are present
        expected_paths = [temp_path / f"{base_name}-{i:05d}-of-{total_parts:05d}.safetensors" for i in range(1, total_parts + 1)]
        assert set(paths) == set(expected_paths)


def test_transformer_model_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        transformer_dir = temp_path / "transformer"
        transformer_dir.mkdir()
        transformer_file = transformer_dir / "diffusion_pytorch_model.safetensors"
        transformer_file.touch()

        paths = get_checkpoint_paths(temp_path)
        assert transformer_file in paths


def test_mixed_files_sharded_checkpoints():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Scenario 5: Mixed files and sharded checkpoints
        mixed_dir = temp_path / "mixed"
        mixed_dir.mkdir()

        # Create a single file
        (mixed_dir / "single_model.safetensors").touch()

        # Create sharded checkpoint
        base_name = "diffusion_pytorch_model"
        total_parts = 2
        for i in range(1, total_parts + 1):
            (mixed_dir / f"{base_name}-{i:05d}-of-{total_parts:05d}.safetensors").touch()

        paths = get_checkpoint_paths(mixed_dir)
        assert len(paths) == total_parts + 1

        # Verify correct handling of Path and str inputs
        path_input = mixed_dir
        str_input = str(mixed_dir)

        path_paths = get_checkpoint_paths(path_input)
        str_paths = get_checkpoint_paths(str_input)

        assert set(path_paths) == set(str_paths)
