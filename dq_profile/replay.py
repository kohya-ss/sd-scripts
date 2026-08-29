from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional

import numpy as np
import torch

from dq_profile.protocol import deterministic_seed


def clone_collated_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone_collated_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_collated_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_collated_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def move_tensors(value: Any, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(value, torch.Tensor):
        target_dtype = dtype if dtype is not None and value.is_floating_point() else value.dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, dict):
        return {key: move_tensors(item, device, dtype=None) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device, dtype=None) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device, dtype=None) for item in value)
    return copy.deepcopy(value)


def _tensor_digest(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        hasher.update(str(value.dtype).encode())
        hasher.update(str(value.shape).encode())
        hasher.update(value.tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            hasher.update(str(key).encode("utf-8", errors="replace"))
            _tensor_digest(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _tensor_digest(hasher, item)
    else:
        hasher.update(repr(value).encode("utf-8", errors="replace"))


def replay_digest(value: Any) -> str:
    hasher = hashlib.sha256()
    _tensor_digest(hasher, value)
    return hasher.hexdigest()


def seed_step_rng(
    protocol_seed: int,
    step_id: str | int,
    phase: str = "branch",
    repeat: int = 0,
) -> int:
    seed = deterministic_seed(
        protocol_seed,
        phase=phase,
        probe_or_step=step_id,
        module_name="__model_rng__",
        invocation=0,
        repeat=int(repeat),
    )
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@dataclass
class ReplayBatch:
    index: int
    source_epoch: int
    source_step: int
    global_step: int
    batch: dict[str, Any]
    latents: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    noisy_latents: Optional[torch.Tensor] = None
    timesteps: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    huber_c: Optional[torch.Tensor | float] = None
    model_seed: Optional[int] = None
    batch_digest: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        self.batch = clone_collated_to_cpu(self.batch)
        for name in ("latents", "noise", "noisy_latents", "timesteps", "target"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.detach().to("cpu").clone())
        if isinstance(self.huber_c, torch.Tensor):
            self.huber_c = self.huber_c.detach().to("cpu").clone()
        self.batch_digest = replay_digest(self.batch)
        self.refresh_digest()

    def refresh_digest(self) -> None:
        materialized = {
            "batch": self.batch,
            "latents": self.latents,
            "noise": self.noise,
            "noisy_latents": self.noisy_latents,
            "timesteps": self.timesteps,
            "target": self.target,
            "huber_c": self.huber_c,
            "model_seed": self.model_seed,
        }
        if all(materialized[name] is None for name in ("latents", "noise", "noisy_latents", "timesteps", "target")):
            self.digest = self.batch_digest
        else:
            self.digest = replay_digest(materialized)

    @property
    def image_keys(self) -> tuple[str, ...]:
        raw = self.batch.get("image_keys", ())
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, (list, tuple)):
            return tuple(str(item) for item in raw)
        return ()

    def materialized(self) -> bool:
        return all(value is not None for value in (self.latents, self.noise, self.noisy_latents, self.timesteps, self.target))

    def runtime_batch(self, device: torch.device | str) -> dict[str, Any]:
        return move_tensors(self.batch, device)


class ReplaySequence:
    """A sealed post-collation sequence; branches cannot access a DataLoader."""

    def __init__(self) -> None:
        self._items: list[ReplayBatch] = []
        self._sealed = False

    def append(self, item: ReplayBatch) -> None:
        if self._sealed:
            raise RuntimeError("replay sequence is sealed")
        self._items.append(item)

    def seal(self) -> None:
        if not self._items:
            raise ValueError("cannot seal an empty replay sequence")
        self._sealed = True

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ReplayBatch]:
        if not self._sealed:
            raise RuntimeError("replay sequence must be sealed before branch use")
        return iter(self._items)

    def __getitem__(self, index: int) -> ReplayBatch:
        if not self._sealed:
            raise RuntimeError("replay sequence must be sealed before branch use")
        return self._items[index]

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "index": item.index,
                "source_epoch": item.source_epoch,
                "source_step": item.source_step,
                "global_step": item.global_step,
                "image_keys": list(item.image_keys),
                "batch_digest": item.batch_digest,
                "digest": item.digest,
                "model_seed": item.model_seed,
                "timestep": None if item.timesteps is None else [int(value) for value in item.timesteps.reshape(-1).tolist()],
            }
            for item in self._items
        ]

    def unique_image_items(self, limit: int) -> list[ReplayBatch]:
        selected: list[ReplayBatch] = []
        seen: set[str] = set()
        for item in self._items:
            key = item.image_keys[0] if item.image_keys else item.digest
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected
