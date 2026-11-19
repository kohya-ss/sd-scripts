import functools
import gc
from typing import Optional, Union

import torch


try:
    # intel gpu support for pytorch older than 2.5
    # ipex is not needed after pytorch 2.5
    import intel_extension_for_pytorch as ipex  # noqa
except Exception:
    pass


try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_MPS = torch.backends.mps.is_available()
except Exception:
    HAS_MPS = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False


def clean_memory():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    if HAS_XPU:
        torch.xpu.empty_cache()
    if HAS_MPS:
        torch.mps.empty_cache()


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_device_descriptions() -> list[dict]:
    r"""
    Returns the accelerator devices (CUDA / XPU / MPS) as a list of `{"device", "name", "total_memory"}` for logging.
    `device` is the device string in PyTorch such as "cuda:0". `total_memory` is in bytes, None if it is unknown.
    """
    devices = []
    if HAS_CUDA:
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                devices.append({"device": f"cuda:{i}", "name": props.name, "total_memory": props.total_memory})
            except Exception:
                pass
    if HAS_XPU:
        for i in range(torch.xpu.device_count()):
            try:
                props = torch.xpu.get_device_properties(i)
                total_memory = getattr(props, "total_memory", None)
                devices.append({"device": f"xpu:{i}", "name": torch.xpu.get_device_name(i), "total_memory": total_memory})
            except Exception:
                pass
    if HAS_MPS:
        try:
            total_memory = torch.mps.recommended_max_memory()
        except Exception:
            total_memory = None
        devices.append({"device": "mps", "name": "Apple MPS", "total_memory": total_memory})
    return devices


def get_device_memory_stats() -> list[dict]:
    r"""
    Returns the memory used by this process on each accelerator device (CUDA / XPU / MPS), as a list of
    `{"device", "allocated", "reserved", "peak_allocated"}` in bytes. A value is None if the backend does not provide it.
    CUDA / XPU devices which this process does not use are omitted. Returns an empty list on CPU.
    """
    stats = []
    for device_type, available in [("cuda", HAS_CUDA), ("xpu", HAS_XPU)]:
        if not available:
            continue
        backend = getattr(torch, device_type)
        try:
            for i in range(backend.device_count()):
                reserved = backend.memory_reserved(i)
                peak = backend.max_memory_allocated(i)
                if reserved == 0 and peak == 0:
                    continue  # this process does not use the device
                stats.append(
                    {
                        "device": f"{device_type}:{i}",
                        "allocated": backend.memory_allocated(i),
                        "reserved": reserved,
                        "peak_allocated": peak,
                    }
                )
        except Exception:
            pass  # the memory functions are not available in this version of the backend
    if HAS_MPS:
        try:
            stats.append(
                {
                    "device": "mps",
                    "allocated": torch.mps.current_allocated_memory(),
                    "reserved": torch.mps.driver_allocated_memory(),
                    "peak_allocated": None,  # MPS does not track the peak
                }
            )
        except Exception:
            pass
    return stats


@functools.lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    r"""
    Do not call this function from training scripts. Use accelerator.device instead.
    """
    if HAS_CUDA:
        device = torch.device("cuda")
    elif HAS_XPU:
        device = torch.device("xpu")
    elif HAS_MPS:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"get_preferred_device() -> {device}")
    return device


def init_ipex():
    """
    Apply IPEX to CUDA hijacks using `library.ipex.ipex_init`.

    This function should run right after importing torch and before doing anything else.

    If xpu is not available, this function does nothing.
    """
    try:
        if HAS_XPU:
            from library.ipex import ipex_init

            is_initialized, error_message = ipex_init()
            if not is_initialized:
                print("failed to initialize ipex:", error_message)
        else:
            return
    except Exception as e:
        print("failed to initialize ipex:", e)
