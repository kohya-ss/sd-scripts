"""Tests for library/safetensors_utils.py (TensorWeightAdapter hooks)."""

import torch
from safetensors.torch import save_file

from library.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks


def _split_rows(key, tensor):
    if not key.endswith(".qkv.weight"):
        return None, None
    base = key[: -len("qkv.weight")]
    keys = [base + c + ".weight" for c in "qkv"]
    if tensor is None:
        return keys, None
    return keys, [t.contiguous() for t in torch.split(tensor, tensor.shape[0] // 3, dim=0)]


def test_split_hook_loads_the_original_tensor_once(tmp_path):
    path = tmp_path / "fused.safetensors"
    fused = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    save_file({"blocks.0.qkv.weight": fused, "blocks.0.norm.weight": torch.ones(4)}, str(path))

    hooks = WeightTransformHooks()
    hooks.split_hook = _split_rows
    with MemoryEfficientSafeOpen(str(path)) as original:
        loads = []
        original_get_tensor = original.get_tensor
        original.get_tensor = lambda key, device=None, dtype=None: loads.append(key) or original_get_tensor(key, device=device, dtype=dtype)
        adapter = TensorWeightAdapter(hooks, original)

        assert set(adapter.keys()) == {"blocks.0.q.weight", "blocks.0.k.weight", "blocks.0.v.weight", "blocks.0.norm.weight"}
        pieces = {key: adapter.get_tensor(key) for key in adapter.keys()}

    # the fused tensor is read and split once; every piece is served from the cache and the cache is emptied
    assert loads.count("blocks.0.qkv.weight") == 1
    assert adapter.tensor_cache == {}
    assert torch.equal(pieces["blocks.0.q.weight"], fused[:4])
    assert torch.equal(pieces["blocks.0.k.weight"], fused[4:8])
    assert torch.equal(pieces["blocks.0.v.weight"], fused[8:])
    assert torch.equal(pieces["blocks.0.norm.weight"], torch.ones(4))
