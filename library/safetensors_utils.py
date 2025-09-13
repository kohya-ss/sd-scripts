import os
import re
import numpy as np
import torch
import json
import struct
from typing import Dict, Any, Union, Optional

from safetensors.torch import load_file

from library.device_utils import synchronize_device


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
    """

    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated


    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    """Memory-efficient reader for safetensors files.

    This class provides a memory-efficient way to read tensors from safetensors files
    by using memory mapping for large tensors and avoiding unnecessary copies.
    """

    def __init__(self, filename):
        """Initialize the SafeTensor reader.

        Args:
            filename (str): Path to the safetensors file to read.
        """
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close file."""
        self.file.close()

    def keys(self):
        """Get all tensor keys in the file.

        Returns:
            list: List of tensor names (excludes metadata).
        """
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """Get metadata from the file.

        Returns:
            Dict[str, str]: Metadata dictionary.
        """
        return self.header.get("__metadata__", {})

    def _read_header(self):
        """Read and parse the header from the safetensors file.

        Returns:
            tuple: (header_dict, header_size) containing parsed header and its size.
        """
        # Read header size (8 bytes, little-endian unsigned long long)
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        # Read and decode header JSON
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a tensor from the file with memory-efficient strategies.

        **Note:**
        If device is 'cuda' , the transfer to GPU is done efficiently using pinned memory and non-blocking transfer.
        So you must ensure that the transfer is completed before using the tensor (e.g., by `torch.cuda.synchronize()`).

        If the tensor is large (>10MB) and the target device is CUDA, memory mapping with numpy.memmap is used to avoid intermediate copies.

        Args:
            key (str): Name of the tensor to load.
            device (Optional[torch.device]): Target device for the tensor.
            dtype (Optional[torch.dtype]): Target dtype for the tensor.

        Returns:
            torch.Tensor: The loaded tensor.

        Raises:
            KeyError: If the tensor key is not found in the file.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        # Handle empty tensors
        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        # Determine if we should use pinned memory for GPU transfer
        non_blocking = device is not None and device.type == "cuda"

        # Calculate absolute file offset
        tensor_offset = self.header_size + 8 + offset_start  # adjust offset by header size

        # Memory mapping strategy for large tensors to GPU
        # Use memmap for large tensors to avoid intermediate copies.
        # If device is cpu, tensor is not copied to gpu, so using memmap locks the file, which is not desired.
        # So we only use memmap if device is not cpu.
        if num_bytes > 10 * 1024 * 1024 and device is not None and device.type != "cpu":
            # Create memory map for zero-copy reading
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            byte_tensor = torch.from_numpy(mm)  # zero copy
            del mm

            # Deserialize tensor (view and reshape)
            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)  # view and reshape
            del byte_tensor

            # Transfer to target device and dtype
            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        # Standard file reading strategy for smaller tensors or CPU target
        # seek to the specified position
        self.file.seek(tensor_offset)

        # read directly into a numpy array by numpy.fromfile without intermediate copy
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array

        # deserialize (view and reshape)
        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        # cast to target dtype and move to device
        return deserialized_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        """Deserialize byte tensor to the correct shape and dtype.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor from file.
            metadata (Dict): Tensor metadata containing dtype and shape info.

        Returns:
            torch.Tensor: Deserialized tensor with correct shape and dtype.
        """
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        # Handle special float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # Standard conversion: view as target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        """Convert string dtype to PyTorch dtype.

        Args:
            dtype_str (str): String representation of the dtype.

        Returns:
            torch.dtype: Corresponding PyTorch dtype.
        """
        # Standard dtype mappings
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # Add float8 types if available in PyTorch version
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        """Convert byte tensor to float8 format if supported.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor.
            dtype_str (str): Float8 dtype string ("F8_E5M2" or "F8_E4M3").
            shape (tuple): Target tensor shape.

        Returns:
            torch.Tensor: Tensor with float8 dtype.

        Raises:
            ValueError: If float8 type is not supported in current PyTorch version.
        """
        # Convert to specific float8 types if available
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # Float8 not supported in this PyTorch version
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


def load_safetensors(
    path: str, device: Union[str, torch.device], disable_mmap: bool = False, dtype: Optional[torch.dtype] = None
) -> dict[str, torch.Tensor]:
    if disable_mmap:
        # return safetensors.torch.load(open(path, "rb").read())
        # use experimental loader
        # logger.info(f"Loading without mmap (experimental)")
        state_dict = {}
        device = torch.device(device) if device is not None else None
        with MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key, device=device, dtype=dtype)
        synchronize_device(device)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)
        except:
            state_dict = load_file(path)  # prevent device invalid Error
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict


def load_split_weights(
    file_path: str, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False, dtype: Optional[torch.dtype] = None
) -> Dict[str, torch.Tensor]:
    """
    Load split weights from a file. If the file name ends with 00001-of-00004 etc, it will load all files with the same prefix.
    dtype is as is, no conversion is done.
    """
    device = torch.device(device)

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        state_dict = {}
        for i in range(count):
            filename = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                state_dict.update(load_safetensors(filepath, device=device, disable_mmap=disable_mmap, dtype=dtype))
            else:
                raise FileNotFoundError(f"File {filepath} not found")
    else:
        state_dict = load_safetensors(file_path, device=device, disable_mmap=disable_mmap, dtype=dtype)
    return state_dict


def find_key(safetensors_file: str, starts_with: Optional[str] = None, ends_with: Optional[str] = None) -> Optional[str]:
    """
    Find a key in a safetensors file that starts with `starts_with` and ends with `ends_with`.
    If `starts_with` is None, it will match any key.
    If `ends_with` is None, it will match any key.
    Returns the first matching key or None if no key matches.
    """
    with MemoryEfficientSafeOpen(safetensors_file) as f:
        for key in f.keys():
            if (starts_with is None or key.startswith(starts_with)) and (ends_with is None or key.endswith(ends_with)):
                return key
    return None
