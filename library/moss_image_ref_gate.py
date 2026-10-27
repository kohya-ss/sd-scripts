# Learned reference gate for the one-way (KV cache) reference editing of MOSS-Image (Step C).
#
# A table of logits a[block, timestep band] gives the gate g = sigmoid(a) with which each DiT block attends to the cached
# reference K/V in each band of the denoising schedule. The gate enters the attention as an additive bias log(g) on the
# logits of the reference keys (see `moss_image_models.gated_reference_attention`), so g = 1 is the plain concatenation
# of the reference keys and g = 0 removes the reference from the block. A budget penalty per band sparsifies the table,
# which is then binarized (top-k blocks per band) for a static, cheaper inference: blocks with g = 0 skip the reference
# (plain attention), blocks with g = 1 use the plain concatenation (no bias dimension, any attention kernel).
#
# The timestep bands are given by their upper boundaries in sigma * 1000 (descending), e.g. "950,850,700,500" gives the
# five bands (1000, 950], (950, 850], (850, 700], (700, 500], (500, 0] which follow the quantization points of the
# inference schedule (50 steps, shift 5).

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BAND_EDGES = "950,850,700,500"
BINARIZED_LOGIT = 20.0  # sigmoid(20) = 1 - 2e-9: the gate is treated as exactly 0 / 1 (see `resolve_block_gates`)
OFF_LOG_GATE = -1.0e4  # log gate of a closed block in the bias path (exp underflows to 0 in float32 softmax)


def parse_band_edges(value: Optional[str]) -> List[float]:
    """"950,850,700,500" -> [950.0, 850.0, 700.0, 500.0] (descending, strictly inside (0, 1000))"""
    if value is None or not str(value).strip():
        value = DEFAULT_BAND_EDGES
    edges = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    edges = sorted(edges, reverse=True)
    if any(not (0.0 < e < 1000.0) for e in edges) or len(set(edges)) != len(edges):
        raise ValueError(f"band edges must be distinct values in (0, 1000): {value}")
    return edges


class ReferenceGate(nn.Module):
    """Gate table a[num_blocks, num_bands]. `trainable=False` keeps the table as a buffer (binarized / frozen table)."""

    def __init__(self, num_blocks: int, band_edges: Sequence[float], init_logit: float = 2.0, trainable: bool = True):
        super().__init__()
        self.num_blocks = num_blocks
        self.register_buffer("band_edges", torch.tensor(sorted(band_edges, reverse=True), dtype=torch.float32))
        logits = torch.full((num_blocks, len(band_edges) + 1), float(init_logit), dtype=torch.float32)
        if trainable:
            self.logits = nn.Parameter(logits)
        else:
            self.register_buffer("logits", logits)

    @property
    def num_bands(self) -> int:
        return self.logits.shape[1]

    @property
    def trainable(self) -> bool:
        return isinstance(self.logits, nn.Parameter)

    def band_index(self, timesteps: torch.Tensor) -> torch.Tensor:
        """[B] timesteps (sigma * 1000) -> [B] band indices: 0 for the highest noise band (t > edges[0])"""
        edges = self.band_edges.to(timesteps.device)
        return (timesteps.float()[:, None] <= edges[None, :]).sum(1)

    def band_ranges(self) -> List[Tuple[float, float]]:
        """(low, high] boundaries of the bands in sigma * 1000, highest noise first"""
        bounds = [1000.0] + self.band_edges.tolist() + [0.0]
        return [(bounds[i + 1], bounds[i]) for i in range(len(bounds) - 1)]

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def log_gates(self, timesteps: torch.Tensor) -> torch.Tensor:
        """[B] timesteps -> [B, num_blocks] log gates of the band of each sample"""
        band = self.band_index(timesteps).to(self.logits.device)  # the table may live on the CPU (inference)
        return F.logsigmoid(self.logits.float())[:, band].transpose(0, 1).to(timesteps.device)

    def budget_penalty(self, budget: float) -> torch.Tensor:
        """Hinge penalty on the number of open blocks per band: mean over bands of relu(sum_blocks g - budget)"""
        return F.relu(self.gates().sum(0) - budget).mean()

    @torch.no_grad()
    def binarize_(self, budget: int):
        """Keep the top `budget` blocks of each band (logit +-BINARIZED_LOGIT)"""
        logits = self.logits.detach()
        top = logits.topk(min(int(budget), self.num_blocks), dim=0).indices  # [budget, num_bands]
        new = torch.full_like(logits, -BINARIZED_LOGIT)
        new.scatter_(0, top, BINARIZED_LOGIT)
        self.logits.copy_(new)

    def is_binarized(self) -> bool:
        return bool((self.logits.detach().abs() >= BINARIZED_LOGIT - 1e-3).all())

    def open_blocks(self, band: int, threshold: float = 0.5) -> List[int]:
        return [i for i, g in enumerate(self.gates()[:, band].tolist()) if g > threshold]

    def table_string(self) -> str:
        """Human readable map: rows = blocks, columns = bands (highest noise first)"""
        gates = self.gates().detach().cpu()
        ranges = self.band_ranges()
        header = "block | " + " | ".join(f"{int(hi):4d}-{int(lo):<4d}" for lo, hi in ranges)
        lines = [header, "-" * len(header)]
        for i in range(self.num_blocks):
            lines.append(f"{i:5d} | " + " | ".join(f"{g:9.3f}" for g in gates[i].tolist()))
        lines.append("-" * len(header))
        lines.append(" open | " + " | ".join(f"{(gates[:, b] > 0.5).sum().item():9d}" for b in range(self.num_bands)))
        lines.append("  sum | " + " | ".join(f"{gates[:, b].sum().item():9.2f}" for b in range(self.num_bands)))
        return "\n".join(lines)


def resolve_block_gates(
    ref_kv: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]], log_gates: Optional[torch.Tensor]
) -> Tuple[List[Optional[Tuple[torch.Tensor, torch.Tensor]]], Optional[List[Optional[torch.Tensor]]]]:
    """Per block, decide how the reference K/V enter the attention from the log gates [B, num_blocks]:
    - None entry in ref_kv, or all samples closed (g ~ 0): no reference in the block (plain attention)
    - all samples fully open (g = 1): plain concatenation of the reference keys (no bias dimension)
    - otherwise: the bias path with the per-sample log gates

    Returns (ref_kv with None for closed blocks, per-block list of log gates for the bias path or None if no block needs it).
    """
    if log_gates is None:
        return list(ref_kv), None
    out_kv: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    out_gates: List[Optional[torch.Tensor]] = []
    closed = F.logsigmoid(torch.tensor(-BINARIZED_LOGIT + 1e-3)).item()  # log gate of a binarized closed block
    for i, kv in enumerate(ref_kv):
        g = log_gates[:, i]
        if kv is None or bool((g <= closed).all()):
            out_kv.append(None)
            out_gates.append(None)
        elif bool((g >= -1e-6).all()):
            out_kv.append(kv)
            out_gates.append(None)
        else:
            out_kv.append(kv)
            out_gates.append(g.clamp(min=OFF_LOG_GATE))
    if all(g is None for g in out_gates):
        out_gates = None
    return out_kv, out_gates
