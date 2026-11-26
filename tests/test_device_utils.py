import types

import pytest

from library import device_utils


@pytest.fixture
def no_devices(monkeypatch):
    for name in ["HAS_CUDA", "HAS_XPU", "HAS_MPS"]:
        monkeypatch.setattr(device_utils, name, False)


def make_backend(memory: dict, names: dict):
    """A fake of `torch.cuda` / `torch.xpu`. `memory` is {index: (allocated, reserved, peak)}."""
    return types.SimpleNamespace(
        device_count=lambda: len(names),
        memory_allocated=lambda i: memory[i][0],
        memory_reserved=lambda i: memory[i][1],
        max_memory_allocated=lambda i: memory[i][2],
        get_device_name=lambda i: names[i],
        get_device_properties=lambda i: types.SimpleNamespace(name=names[i], total_memory=1000 * (i + 1)),
    )


def test_cpu_only(no_devices):
    assert device_utils.get_device_memory_stats() == []
    assert device_utils.get_device_descriptions() == []


@pytest.mark.parametrize("device_type, flag", [("cuda", "HAS_CUDA"), ("xpu", "HAS_XPU")])
def test_cuda_and_xpu(no_devices, monkeypatch, device_type, flag):
    backend = make_backend({0: (10, 20, 30), 1: (0, 0, 0)}, {0: "GPU A", 1: "GPU B"})
    monkeypatch.setattr(device_utils, flag, True)
    monkeypatch.setattr(device_utils.torch, device_type, backend)

    # the device which this process does not use is omitted
    assert device_utils.get_device_memory_stats() == [
        {"device": f"{device_type}:0", "allocated": 10, "reserved": 20, "peak_allocated": 30}
    ]
    assert device_utils.get_device_descriptions() == [
        {"device": f"{device_type}:0", "name": "GPU A", "total_memory": 1000},
        {"device": f"{device_type}:1", "name": "GPU B", "total_memory": 2000},
    ]


def test_backend_without_memory_functions(no_devices, monkeypatch):
    # e.g. an old version of the XPU backend
    monkeypatch.setattr(device_utils, "HAS_XPU", True)
    monkeypatch.setattr(device_utils.torch, "xpu", types.SimpleNamespace(device_count=lambda: 1))
    assert device_utils.get_device_memory_stats() == []
    assert device_utils.get_device_descriptions() == []


def test_set_vram_limit_fraction(no_devices, monkeypatch):
    calls = []
    backend = types.SimpleNamespace(current_device=lambda: 1, set_per_process_memory_fraction=lambda f, i: calls.append((f, i)))
    monkeypatch.setattr(device_utils.torch, "cuda", backend)
    devices = [device_utils.torch.device("cuda:0"), "cuda:2"]

    # not specified, or no CUDA device: nothing is done
    monkeypatch.setattr(device_utils, "HAS_CUDA", True)
    assert device_utils.set_vram_limit_fraction(None, devices) == []
    monkeypatch.setattr(device_utils, "HAS_CUDA", False)
    assert device_utils.set_vram_limit_fraction(0.95, devices) == []
    assert calls == []

    # only the given CUDA devices are limited
    monkeypatch.setattr(device_utils, "HAS_CUDA", True)
    assert device_utils.set_vram_limit_fraction(0.95, devices) == ["cuda:0", "cuda:2"]
    assert calls == [(0.95, 0), (0.95, 2)]
    assert device_utils.set_vram_limit_fraction(1.0, ["cpu", "mps"]) == []

    # "cuda" without an index is the current device
    assert device_utils.set_vram_limit_fraction(1.0, ["cuda"]) == ["cuda:1"]
    assert calls[-1] == (1.0, 1)


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.01, 95])
def test_set_vram_limit_fraction_invalid(no_devices, fraction):
    # the value is checked even without a CUDA device
    with pytest.raises(ValueError):
        device_utils.set_vram_limit_fraction(fraction, ["cuda:0"])


def test_mps(no_devices, monkeypatch):
    mps = types.SimpleNamespace(
        current_allocated_memory=lambda: 10, driver_allocated_memory=lambda: 20, recommended_max_memory=lambda: 1000
    )
    monkeypatch.setattr(device_utils, "HAS_MPS", True)
    monkeypatch.setattr(device_utils.torch, "mps", mps)

    assert device_utils.get_device_memory_stats() == [{"device": "mps", "allocated": 10, "reserved": 20, "peak_allocated": None}]
    assert device_utils.get_device_descriptions() == [{"device": "mps", "name": "Apple MPS", "total_memory": 1000}]

    # recommended_max_memory does not exist in old versions
    monkeypatch.setattr(device_utils.torch, "mps", types.SimpleNamespace())
    assert device_utils.get_device_memory_stats() == []
    assert device_utils.get_device_descriptions() == [{"device": "mps", "name": "Apple MPS", "total_memory": None}]
