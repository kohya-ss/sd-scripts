"""Local-only regression harness for transformers / diffusers upgrades.

This script is NOT run by CI. It needs locally available model weights (paths
are given in ``tests/local/models.toml``, which is git-ignored) and, on the
first run, the tokenizers from the Hugging Face Hub (cached afterwards, so
later runs work with ``HF_HUB_OFFLINE=1``).

Workflow:

1. With the *current* dependency versions, record reference outputs::

       python tests/local/regression_te.py record

2. Upgrade transformers / diffusers, then compare against the references::

       python tests/local/regression_te.py compare
       # or, as pytest:  pytest tests/local

What is recorded, per model listed in the config:

* text encoders: the exact objects the training scripts use
  (``TokenizeStrategy.tokenize`` -> ``TextEncodingStrategy.encode_tokens``,
  text encoders built by the family's own loader), for a fixed set of prompts,
  one prompt at a time and once as a batch (padding + attention masks);
* for SD / SDXL: encode / decode of a fixed image through the diffusers
  ``AutoencoderKL`` that the checkpoint loader builds;
* always: the diffusers noise schedulers used for sample generation
  (timesteps, alphas_cumprod, sigmas, and a few ``step()`` calls with fixed
  noise).

Token ids and masks must match exactly. Floating point outputs are compared
with ``numpy.allclose`` using per-model tolerances (see ``models.example.toml``)
and the maximum absolute / relative difference and cosine similarity are
printed for every tensor so that small drifts stay visible.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import inspect
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import toml
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = LOCAL_DIR / "models.toml"
DEFAULT_REFERENCE_DIR = LOCAL_DIR / "references"

SCHEDULERS_NAME = "schedulers"

# Fixed prompts. Keep this list stable: changing it invalidates recorded references.
PROMPTS: List[str] = [
    "",  # empty (unconditional)
    "a photo of a cat",
    "1girl, solo, long hair, looking at viewer, smile, blue eyes, white shirt, outdoors, cherry blossoms, masterpiece, best quality",
    # Japanese + emoji: exercises byte-level / unicode handling of the tokenizers
    "桜の木の下で本を読む少女、夕暮れ、水彩画風🌸📖",
    # quoted text: HunyuanImage extracts it for the byT5 glyph encoder (ASCII and CJK quotes)
    'A neon sign on a brick wall that says "OPEN 24 HOURS", with a small paper poster below reading “本日開店”',
    # ~100 CLIP tokens: crosses the 77-token boundary (SD / SDXL chunking)
    (
        "A highly detailed oil painting of an ancient library carved into the side of a mountain, "
        "with towering wooden shelves, thousands of leather-bound books, warm candle light flickering "
        "on the stone walls, dust particles floating in shafts of golden sunlight coming through tall "
        "arched windows, a spiral staircase in the center, intricate carvings of owls and vines on the "
        "railings, a single scholar in a dark green robe reading at a long oak table"
    ),
    # ~350 CLIP tokens: exceeds max_token_length=225 as well, so truncation is exercised too
    (
        "Wide establishing shot of a futuristic coastal city at dawn: sleek white towers with hanging "
        "gardens, transparent skybridges connecting them, small delivery drones drifting between the "
        "buildings, a monorail curving along the shoreline, fishing boats with colorful sails leaving "
        "the harbor, mist rising from the water, warm orange light on the east-facing facades and cool "
        "blue shadows on the west, seagulls, a lighthouse on a rocky promontory, cargo ships on the "
        "horizon, reflections in the wet streets, a few early commuters with umbrellas, street vendors "
        "setting up stalls with fruit and flowers, tram wires overhead, ivy on old brick warehouses "
        "converted into cafes, bicycles leaning against lamp posts, a large digital billboard showing "
        "the weather forecast, a park with cherry trees in full bloom, children playing near a fountain, "
        "an old man feeding pigeons, a dog chasing a ball, detailed textures, photorealistic, 8k, "
        "cinematic lighting, shallow depth of field, shot on 35mm film, subtle grain"
    ),
]

SAMPLERS = [
    "ddim",
    "ddpm",
    "pndm",
    "lms",
    "euler",
    "euler_a",
    "dpmsolver",
    "dpmsolver++",
    "dpmsingle",
    "heun",
    "dpm_2",
    "dpm_2_a",
]

DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

# default tolerances (atol, rtol) by dtype of the text encoder computation
DEFAULT_TOLERANCES = {
    torch.float32: (1e-4, 1e-3),
    torch.float16: (2e-2, 2e-2),
    torch.bfloat16: (5e-2, 3e-2),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.numpy()
    if isinstance(x, (int, float, bool)):
        return np.asarray(x)
    if isinstance(x, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in x):
        return np.stack([_to_numpy(t) for t in x])
    raise TypeError(f"unsupported output type: {type(x)}")


def _store(out: Dict[str, np.ndarray], prefix: str, values: Any) -> None:
    """Flatten a list/tuple of tensors (or a single tensor) into ``out``."""
    if isinstance(values, (list, tuple)):
        for i, v in enumerate(values):
            if v is None:
                continue
            out[f"{prefix}.{i}"] = _to_numpy(v)
    elif values is not None:
        out[prefix] = _to_numpy(values)


def _versions() -> Dict[str, str]:
    import diffusers
    import transformers

    try:
        import huggingface_hub

        hub = huggingface_hub.__version__
    except Exception:  # pragma: no cover
        hub = "?"
    try:
        import accelerate

        acc = accelerate.__version__
    except Exception:  # pragma: no cover
        acc = "?"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
        "accelerate": acc,
        "huggingface_hub": hub,
        "platform": f"{platform.system()} {platform.machine()}",
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _fixed_image(size: int = 512, device: str = "cpu") -> torch.Tensor:
    """Deterministic smooth-ish RGB image in [-1, 1], shape [1, 3, size, size]."""
    rng = np.random.RandomState(0)
    low = torch.from_numpy(rng.rand(1, 3, 32, 32).astype(np.float32))
    img = torch.nn.functional.interpolate(low, size=(size, size), mode="bicubic", align_corners=False)
    # add a little high-frequency detail so the encoder sees more than a blur
    noise = torch.from_numpy(rng.rand(1, 3, size, size).astype(np.float32)) * 0.05
    img = (img + noise).clamp(0, 1) * 2 - 1
    return img.to(device)


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# generic text encoder recording
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_prompts(tokenize_strategy, encoding_strategy, models: List[Any], encode_kwargs: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Run tokenize -> encode_tokens for every prompt (batch size 1) and once for all prompts as a batch."""
    encode_kwargs = encode_kwargs or {}
    out: Dict[str, np.ndarray] = {}
    for i, prompt in enumerate(PROMPTS):
        tokens = tokenize_strategy.tokenize(prompt)
        _store(out, f"p{i}.tok", tokens)
        encoded = encoding_strategy.encode_tokens(tokenize_strategy, models, tokens, **encode_kwargs)
        _store(out, f"p{i}.out", encoded)

    tokens = tokenize_strategy.tokenize(list(PROMPTS))
    _store(out, "batch.tok", tokens)
    encoded = encoding_strategy.encode_tokens(tokenize_strategy, models, tokens, **encode_kwargs)
    _store(out, "batch.out", encoded)
    return out


@torch.no_grad()
def encode_vae(vae, device: str) -> Dict[str, np.ndarray]:
    """Encode / decode a fixed image with a diffusers AutoencoderKL (fp32)."""
    vae = vae.to(device=device, dtype=torch.float32)
    img = _fixed_image(512, device)
    dist = vae.encode(img).latent_dist
    decoded = vae.decode(dist.mean).sample
    return {
        "vae.latent_mean": _to_numpy(dist.mean),
        "vae.latent_std": _to_numpy(dist.std),
        "vae.decoded": _to_numpy(decoded),
    }


# ---------------------------------------------------------------------------
# model families
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    type: str
    cfg: Dict[str, Any]
    device: str
    dtype: torch.dtype
    atol: float
    rtol: float

    @staticmethod
    def from_config(entry: Dict[str, Any], common: Dict[str, Any]) -> "ModelSpec":
        name = entry["name"]
        mtype = entry["type"]
        device = entry.get("device", common.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        default_dtype = "float32" if mtype in ("sd", "sdxl") else "bfloat16"
        dtype = DTYPES[str(entry.get("dtype", default_dtype)).lower()]
        d_atol, d_rtol = DEFAULT_TOLERANCES[dtype]
        return ModelSpec(
            name=name,
            type=mtype,
            cfg=entry,
            device=device,
            dtype=dtype,
            atol=float(entry.get("atol", d_atol)),
            rtol=float(entry.get("rtol", d_rtol)),
        )


def run_sd(spec: ModelSpec) -> Dict[str, np.ndarray]:
    from library import model_util, strategy_sd

    cfg = spec.cfg
    v2 = bool(cfg.get("v2", False))
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(v2, cfg["ckpt"], spec.device, spec.dtype)
    text_encoder.eval()
    tok = strategy_sd.SdTokenizeStrategy(v2, cfg.get("max_token_length"), cfg.get("tokenizer_cache_dir"))
    enc = strategy_sd.SdTextEncodingStrategy(cfg.get("clip_skip"))
    out = encode_prompts(tok, enc, [text_encoder])
    if cfg.get("check_vae", True):
        out.update(encode_vae(vae, spec.device))
    _free(text_encoder, vae, unet)
    return out


def run_sdxl(spec: ModelSpec) -> Dict[str, np.ndarray]:
    from library import sdxl_model_util, strategy_sdxl

    cfg = spec.cfg
    text_model1, text_model2, vae, unet, _logit_scale, _ckpt_info = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, cfg["ckpt"], spec.device, spec.dtype
    )
    text_model1.to(spec.device).eval()
    text_model2.to(spec.device).eval()
    tok = strategy_sdxl.SdxlTokenizeStrategy(cfg.get("max_token_length"), cfg.get("tokenizer_cache_dir"))
    enc = strategy_sdxl.SdxlTextEncodingStrategy()
    out = encode_prompts(tok, enc, [text_model1, text_model2])
    if cfg.get("check_vae", True):
        out.update(encode_vae(vae, spec.device))
    _free(text_model1, text_model2, vae, unet)
    return out


def run_sd3(spec: ModelSpec) -> Dict[str, np.ndarray]:
    from library import sd3_utils, strategy_sd3

    cfg = spec.cfg
    clip_l = sd3_utils.load_clip_l(cfg["clip_l"], spec.dtype, spec.device)
    clip_g = sd3_utils.load_clip_g(cfg["clip_g"], spec.dtype, spec.device)
    t5xxl = sd3_utils.load_t5xxl(cfg["t5xxl"], spec.dtype, spec.device)
    for m in (clip_l, clip_g, t5xxl):
        # the loaders build the models with init_empty_weights + assign=True; the trainers move them
        # explicitly afterwards, so do the same here
        m.to(spec.device).eval()
    tok = strategy_sd3.Sd3TokenizeStrategy(int(cfg.get("t5xxl_max_token_length", 256)), cfg.get("tokenizer_cache_dir"))
    enc = strategy_sd3.Sd3TextEncodingStrategy(
        apply_lg_attn_mask=bool(cfg.get("apply_lg_attn_mask", False)),
        apply_t5_attn_mask=bool(cfg.get("apply_t5_attn_mask", False)),
    )
    out = encode_prompts(tok, enc, [clip_l, clip_g, t5xxl], {"enable_dropout": False})
    _free(clip_l, clip_g, t5xxl)
    return out


def run_flux(spec: ModelSpec) -> Dict[str, np.ndarray]:
    from library import flux_utils, strategy_flux

    cfg = spec.cfg
    clip_l = flux_utils.load_clip_l(cfg["clip_l"], spec.dtype, spec.device)
    t5xxl = flux_utils.load_t5xxl(cfg["t5xxl"], spec.dtype, spec.device)
    clip_l.to(spec.device).eval()
    t5xxl.to(spec.device).eval()
    tok = strategy_flux.FluxTokenizeStrategy(int(cfg.get("t5xxl_max_token_length", 512)), cfg.get("tokenizer_cache_dir"))
    enc = strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=bool(cfg.get("apply_t5_attn_mask", False)))
    out = encode_prompts(tok, enc, [clip_l, t5xxl])
    _free(clip_l, t5xxl)
    return out


def run_lumina(spec: ModelSpec) -> Dict[str, np.ndarray]:
    from library import lumina_util, strategy_lumina

    cfg = spec.cfg
    gemma2 = lumina_util.load_gemma2(cfg["gemma2"], spec.dtype, spec.device)
    gemma2.to(spec.device).eval()
    tok = strategy_lumina.LuminaTokenizeStrategy(cfg.get("system_prompt"), cfg.get("max_token_length"), cfg.get("tokenizer_cache_dir"))
    enc = strategy_lumina.LuminaTextEncodingStrategy()
    out = encode_prompts(tok, enc, [gemma2])
    _free(gemma2)
    return out


def run_hunyuan_image(spec: ModelSpec) -> Dict[str, np.ndarray]:
    from library import hunyuan_image_text_encoder, strategy_hunyuan_image

    cfg = spec.cfg
    _vlm_tokenizer, qwen2vl = hunyuan_image_text_encoder.load_qwen2_5_vl(cfg["qwen2_5_vl"], spec.dtype, spec.device)
    _byt5_tokenizer, byt5 = hunyuan_image_text_encoder.load_byt5(cfg["byt5"], spec.dtype, spec.device)  # (tokenizer, model)
    qwen2vl.to(spec.device).eval()
    byt5.to(spec.device).eval()
    tok = strategy_hunyuan_image.HunyuanImageTokenizeStrategy(cfg.get("tokenizer_cache_dir"))
    enc = strategy_hunyuan_image.HunyuanImageTextEncodingStrategy()
    out = encode_prompts(tok, enc, [qwen2vl, byt5])
    # The VLM hidden states at padded positions are garbage that the model masks out downstream
    # (encoder_attention_mask); they also depend on the transformers version. Zero them so that
    # only the positions that matter are compared. out.0 = embeddings, out.1 = attention mask.
    for prefix in [f"p{i}" for i in range(len(PROMPTS))] + ["batch"]:
        emb, mask = out.get(f"{prefix}.out.0"), out.get(f"{prefix}.out.1")
        if emb is not None and mask is not None:
            out[f"{prefix}.out.0"] = emb * mask[..., None].astype(emb.dtype)
    _free(qwen2vl, byt5)
    return out


def run_anima(spec: ModelSpec) -> Dict[str, np.ndarray]:
    from library import anima_utils, strategy_anima

    cfg = spec.cfg
    qwen3, _tokenizer = anima_utils.load_qwen3_text_encoder(cfg["qwen3"], spec.dtype, spec.device)
    qwen3.eval()
    tok = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=cfg["qwen3"],
        t5_tokenizer_path=cfg.get("t5_tokenizer_path"),
        qwen3_max_length=int(cfg.get("qwen3_max_token_length", 512)),
        t5_max_length=int(cfg.get("t5_max_token_length", 512)),
    )
    enc = strategy_anima.AnimaTextEncodingStrategy()
    out = encode_prompts(tok, enc, [qwen3])
    _free(qwen3)
    return out


@torch.no_grad()
def run_schedulers(spec: Optional[ModelSpec] = None) -> Dict[str, np.ndarray]:
    """Exercise the diffusers schedulers used for sample image generation (no weights needed)."""
    from library import sampling

    out: Dict[str, np.ndarray] = {}
    for sampler in SAMPLERS:
        for v_param in (False, True):
            key = f"{sampler}.v{int(v_param)}"
            try:
                sched = sampling.get_my_scheduler(sample_sampler=sampler, v_parameterization=v_param)
            except Exception as e:
                # e.g. LMSDiscreteScheduler needs scipy (not in requirements.txt); a scheduler that cannot be
                # built with the installed diffusers is reported and skipped. If it starts working after an
                # upgrade, the new keys show up as "not in reference" -- re-record in that case.
                print(f"    skipping sampler {sampler!r}: {type(e).__name__}: {str(e).strip().splitlines()[0]}")
                break
            sched.set_timesteps(20)
            out[f"{key}.timesteps"] = _to_numpy(torch.as_tensor(sched.timesteps))
            out[f"{key}.alphas_cumprod"] = _to_numpy(sched.alphas_cumprod)
            if getattr(sched, "sigmas", None) is not None:
                out[f"{key}.sigmas"] = _to_numpy(torch.as_tensor(sched.sigmas))

            gen = torch.Generator("cpu").manual_seed(1234)
            sample = torch.randn(1, 4, 8, 8, generator=gen)
            if hasattr(sched, "init_noise_sigma"):
                sample = sample * sched.init_noise_sigma
            step_params = inspect.signature(sched.step).parameters
            for k in range(4):
                t = sched.timesteps[k]
                model_input = sched.scale_model_input(sample, t) if hasattr(sched, "scale_model_input") else sample
                model_output = torch.randn(model_input.shape, generator=gen) * 0.5
                kwargs = {"generator": gen} if "generator" in step_params else {}
                sample = sched.step(model_output, t, sample, **kwargs).prev_sample
            out[f"{key}.sample_after_4_steps"] = _to_numpy(sample)
    return out


RUNNERS: Dict[str, Callable[[ModelSpec], Dict[str, np.ndarray]]] = {
    "sd": run_sd,
    "sdxl": run_sdxl,
    "sd3": run_sd3,
    "flux": run_flux,
    "lumina": run_lumina,
    "hunyuan_image": run_hunyuan_image,
    "anima": run_anima,
}


# ---------------------------------------------------------------------------
# config / references
# ---------------------------------------------------------------------------


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Copy models.example.toml to models.toml and fill in your local model paths.")
    return toml.load(path)


def model_specs(config: Dict[str, Any], include_disabled: bool = False) -> List[ModelSpec]:
    common = config.get("common", {})
    specs = []
    for entry in config.get("models", []):
        if not entry.get("enabled", True) and not include_disabled:
            continue
        if entry["type"] not in RUNNERS:
            raise ValueError(f"unknown model type {entry['type']!r} for {entry['name']!r}; expected one of {sorted(RUNNERS)}")
        specs.append(ModelSpec.from_config(entry, common))
    return specs


def reference_dir(config: Dict[str, Any]) -> Path:
    d = config.get("common", {}).get("reference_dir")
    return Path(d) if d else DEFAULT_REFERENCE_DIR


def schedulers_spec(config: Dict[str, Any]) -> ModelSpec:
    atol, rtol = DEFAULT_TOLERANCES[torch.float32]
    return ModelSpec(name=SCHEDULERS_NAME, type=SCHEDULERS_NAME, cfg={}, device="cpu", dtype=torch.float32, atol=atol, rtol=rtol)


def all_specs(config: Dict[str, Any], names: Optional[List[str]] = None, include_disabled: bool = False) -> List[ModelSpec]:
    """All runnable specs. Entries with ``enabled = false`` are skipped unless they are named explicitly in ``names``."""
    if names:
        # an explicit name overrides `enabled = false`
        specs = [schedulers_spec(config)] + model_specs(config, include_disabled=True)
        missing = set(names) - {s.name for s in specs}
        if missing:
            raise ValueError(f"unknown model name(s): {sorted(missing)}; available: {[s.name for s in specs]}")
        return [s for s in specs if s.name in names]
    return [schedulers_spec(config)] + model_specs(config, include_disabled=include_disabled)


def run_spec(spec: ModelSpec) -> Dict[str, np.ndarray]:
    if spec.type == SCHEDULERS_NAME:
        return run_schedulers(spec)
    return RUNNERS[spec.type](spec)


def record(config: Dict[str, Any], names: Optional[List[str]] = None) -> None:
    ref_dir = reference_dir(config)
    ref_dir.mkdir(parents=True, exist_ok=True)
    for spec in all_specs(config, names):
        print(f"=== record {spec.name} ({spec.type}, device={spec.device}, dtype={spec.dtype})")
        outputs = run_spec(spec)
        np.savez_compressed(ref_dir / f"{spec.name}.npz", **outputs)
        meta = {
            "name": spec.name,
            "type": spec.type,
            "device": spec.device,
            "dtype": str(spec.dtype).replace("torch.", ""),
            "prompts": PROMPTS,
            "keys": sorted(outputs.keys()),
            "versions": _versions(),
        }
        (ref_dir / f"{spec.name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    saved {len(outputs)} arrays -> {ref_dir / (spec.name + '.npz')}")


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


@dataclass
class CompareResult:
    name: str
    ok: bool
    lines: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return "\n".join(self.lines + ([""] + self.failures if self.failures else []))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_arrays(name: str, ref: Dict[str, np.ndarray], cur: Dict[str, np.ndarray], atol: float, rtol: float) -> CompareResult:
    result = CompareResult(name=name, ok=True)
    keys = sorted(set(ref) | set(cur))
    width = max(len(k) for k in keys) if keys else 10
    result.lines.append(f"--- {name}: {len(keys)} arrays, atol={atol:g}, rtol={rtol:g}")
    for k in keys:
        if k not in ref:
            result.failures.append(f"{k}: not in reference (new output)")
            result.ok = False
            continue
        if k not in cur:
            result.failures.append(f"{k}: missing in current outputs")
            result.ok = False
            continue
        r, c = ref[k], cur[k]
        if r.shape != c.shape:
            result.failures.append(f"{k}: shape mismatch reference={r.shape} current={c.shape}")
            result.ok = False
            continue
        if r.dtype.kind in "iub" or c.dtype.kind in "iub":
            n_diff = int(np.count_nonzero(r != c))
            status = "OK" if n_diff == 0 else "MISMATCH"
            result.lines.append(f"{k:<{width}}  {str(r.shape):<22} exact  diff_elems={n_diff:<8} {status}")
            if n_diff:
                result.failures.append(f"{k}: {n_diff} differing integer elements")
                result.ok = False
            continue
        r64, c64 = r.astype(np.float64), c.astype(np.float64)
        finite = np.isfinite(r64) & np.isfinite(c64)
        nan_mismatch = int(np.count_nonzero(np.isfinite(r64) != np.isfinite(c64)))
        diff = np.abs(r64 - c64)[finite]
        max_abs = float(diff.max()) if diff.size else 0.0
        denom = np.abs(r64)[finite]
        max_rel = float((diff / np.maximum(denom, 1e-12)).max()) if diff.size else 0.0
        cos = _cosine(r64[finite], c64[finite]) if diff.size else 1.0
        close = np.allclose(r64[finite], c64[finite], atol=atol, rtol=rtol) and nan_mismatch == 0
        n_bad = int(np.count_nonzero(diff > atol + rtol * denom))
        status = "OK" if close else "MISMATCH"
        result.lines.append(
            f"{k:<{width}}  {str(r.shape):<22} max_abs={max_abs:<10.3e} max_rel={max_rel:<10.3e} cos={cos:.6f} bad={n_bad:<8} {status}"
        )
        if not close:
            result.failures.append(f"{k}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} cos={cos:.6f} bad_elems={n_bad} nan_mismatch={nan_mismatch}")
            result.ok = False
    return result


def compare_spec(spec: ModelSpec, ref_dir: Path) -> CompareResult:
    ref_path = ref_dir / f"{spec.name}.npz"
    if not ref_path.exists():
        raise FileNotFoundError(f"reference {ref_path} not found; run `python tests/local/regression_te.py record --models {spec.name}` first")
    meta_path = ref_dir / f"{spec.name}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("prompts") != PROMPTS:
            raise RuntimeError(f"{spec.name}: PROMPTS changed since the reference was recorded; re-record")
        ref_versions = meta.get("versions", {})
    else:
        ref_versions = {}
    with np.load(ref_path) as f:
        ref = {k: f[k] for k in f.files}
    print(f"=== compare {spec.name} ({spec.type}, device={spec.device}, dtype={spec.dtype})")
    if ref_versions:
        cur_versions = _versions()
        for lib in ("transformers", "diffusers", "torch", "accelerate", "huggingface_hub"):
            print(f"    {lib:<16} reference={ref_versions.get(lib, '?'):<14} current={cur_versions.get(lib, '?')}")
    cur = run_spec(spec)
    return compare_arrays(spec.name, ref, cur, spec.atol, spec.rtol)


def compare(config: Dict[str, Any], names: Optional[List[str]] = None, verbose: bool = True) -> bool:
    ref_dir = reference_dir(config)
    all_ok = True
    for spec in all_specs(config, names):
        result = compare_spec(spec, ref_dir)
        if verbose:
            print(result.summary)
        print(f"    {spec.name}: {'OK' if result.ok else 'FAILED'}")
        all_ok &= result.ok
    print("ALL OK" if all_ok else "SOME COMPARISONS FAILED")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "compare", "list"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to models.toml")
    parser.add_argument("--models", type=str, default=None, help="comma separated model names to process (default: all)")
    parser.add_argument("--quiet", action="store_true", help="compare: print only the per-model verdicts")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    names = [n.strip() for n in args.models.split(",")] if args.models else None

    if args.mode == "list":
        for spec in all_specs(config, include_disabled=True):
            enabled = spec.cfg.get("enabled", True)
            print(
                f"{spec.name:<20} type={spec.type:<14} device={spec.device:<6} dtype={str(spec.dtype).replace('torch.', ''):<9} "
                f"atol={spec.atol:g} rtol={spec.rtol:g}{'' if enabled else '  (enabled = false; runs only when named with --models)'}"
            )
        return 0
    if args.mode == "record":
        record(config, names)
        return 0
    return 0 if compare(config, names, verbose=not args.quiet) else 1


if __name__ == "__main__":
    sys.exit(main())
