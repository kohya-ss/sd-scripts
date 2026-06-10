This file provides the overview and guidance for developers working with the codebase, including setup instructions, architecture details, and common commands.

## Project Architecture

### Core Training Framework
The codebase is built around a **strategy pattern architecture** that supports multiple diffusion model families:

- **`library/strategy_base.py`**: Base classes for tokenization, text encoding, latent caching, and training strategies
- **`library/strategy_*.py`**: Model-specific implementations for SD, SDXL, SD3, FLUX, etc.
- **`library/config_util.py`**: Configuration management with TOML support

### Core Library Modules
The shared training utilities (formerly a single large `train_util.py`) are split into single-responsibility modules:

| Module | Owns |
|---|---|
| `library/dataset.py` | Dataset core: `BaseDataset` / `DatasetGroup` hierarchy, bucketing (`BucketManager`), `ImageInfo`, image globbing/loading, `debug_dataset` |
| `library/subset.py` | Subset definitions: `DreamBoothSubset`, `FineTuningSubset`, `ControlNetSubset` |
| `library/dreambooth_dataset.py` | `DreamBoothDataset` |
| `library/finetuning_dataset.py` | `FineTuningDataset` |
| `library/controlnet_dataset.py` | `ControlNetDataset` |
| `library/caching.py` | Latent / text-encoder-output disk caching (`.npz` read & write helpers) |
| `library/args.py` | All `add_*_arguments` parser registrations, `verify_*` validation, `read_config_from_file` |
| `library/accelerator_setup.py` | `prepare_accelerator`, `prepare_dtype`, `HIGH_VRAM` flag |
| `library/optimizer.py` | `get_optimizer` dispatcher (AdamW / 8-bit / Lion / DAdaptation / Prodigy / Adafactor / schedule-free), LR schedulers |
| `library/model_io.py` | Checkpoint loading (`load_target_model`), model hashes, safetensors metadata, SAI model spec |
| `library/checkpoint_io.py` | Save / rotate checkpoints and train state (`save_sd_model_on_*`, filename templates) |
| `library/loss.py` | Per-step loss building blocks: timestep sampling, noise, `conditional_loss` |
| `library/hidden_states.py` | Text encoder hidden-state extraction (`get_hidden_states`, `get_hidden_states_sdxl`) |
| `library/sampling.py` | Sample image generation during training (`sample_images`, prompt parsing) |
| `library/logging_util.py` | `init_trackers` (TensorBoard / W&B), `LossRecorder` |
| `library/train_util.py` | **Backward-compatibility shim only.** Re-exports the names above for external forks; do not add code here |

Import conventions:
- New code imports from the canonical modules directly (e.g. `from library.dataset import DatasetGroup`), never via `library.train_util`.
- Module-qualified style is `import library.X as X`; where the plain name collides with common local variables, use the `_util` alias: `loss_util`, `optimizer_util`, `args_util`, `dataset_util`.
- `library/train_util.py` is kept only so that external forks importing `library.train_util` keep working. Its re-export list must stay in sync if names move again.

### Model Support Structure
Each supported model family has a consistent structure:
- **Training script**: `{model}_train.py` (full fine-tuning), `{model}_train_network.py` (LoRA/network training)
- **Model utilities**: `library/{model}_models.py`, `library/{model}_train_utils.py`, `library/{model}_utils.py`
- **Networks**: `networks/lora_{model}.py`, `networks/oft_{model}.py` for adapter training

### Supported Models
- **Stable Diffusion 1.x**: `train*.py`, `train_db.py` (for DreamBooth)
- **SDXL**: `sdxl_train*.py`, `library/sdxl_*`
- **SD3**: `sd3_train*.py`, `library/sd3_*`
- **FLUX.1**: `flux_train*.py`, `library/flux_*`
- **Lumina Image 2.0**: `lumina_train*.py`, `library/lumina_*`
- **HunyuanImage-2.1**: `hunyuan_image_train*.py`, `library/hunyuan_image_*`
- **Anima-Preview**:  `anima_train*.py`, `library/anima_*`

### Key Components

#### Memory Management
- **Block swapping**: CPU-GPU memory optimization via `--blocks_to_swap` parameter, works with custom offloading. Only available for models with transformer architectures like SD3 and FLUX.1.
- **Custom offloading**: `library/custom_offloading_utils.py` for advanced memory management
- **Gradient checkpointing**: Memory reduction during training

#### Training Features
- **LoRA training**: Low-rank adaptation networks in `networks/lora*.py`
- **ControlNet training**: Conditional generation control
- **Textual Inversion**: Custom embedding training
- **Multi-resolution training**: Bucket-based aspect ratio handling
- **Validation loss**: Real-time training monitoring, only for LoRA training

#### Configuration System
Dataset configuration uses TOML files with structured validation:
```toml
[datasets.sample_dataset]
  resolution = 1024
  batch_size = 2
  
  [[datasets.sample_dataset.subsets]]
    image_dir = "path/to/images"
    caption_extension = ".txt"
```

## Common Development Commands

### Training Commands Pattern
All training scripts follow this general pattern:
```bash
accelerate launch --mixed_precision bf16 {script_name}.py \
  --pretrained_model_name_or_path model.safetensors \
  --dataset_config config.toml \
  --output_dir output \
  --output_name model_name \
  [model-specific options]
```

### Memory Optimization
For low VRAM environments, use block swapping:
```bash
# Add to any training command for memory reduction
--blocks_to_swap 10  # Swap 10 blocks to CPU (adjust number as needed)
```

### Utility Scripts
Located in `tools/` directory:
- `tools/merge_lora.py`: Merge LoRA weights into base models
- `tools/cache_latents.py`: Pre-cache VAE latents for faster training
- `tools/cache_text_encoder_outputs.py`: Pre-cache text encoder outputs

## Development Notes

### Strategy Pattern Implementation
When adding support for new models, implement the four core strategies:
1. `TokenizeStrategy`: Text tokenization handling
2. `TextEncodingStrategy`: Text encoder forward pass
3. `LatentsCachingStrategy`: VAE encoding/caching
4. `TextEncoderOutputsCachingStrategy`: Text encoder output caching

### Testing Approach
- Unit tests focus on utility functions and model loading
- Integration tests validate training script syntax and basic execution
- Most tests use mocks to avoid requiring actual model files
- Add tests for new model support in `tests/test_{model}_*.py`

### Configuration System
- Use `config_util.py` dataclasses for type-safe configuration
- Support both command-line arguments and TOML file configuration
- Validate configuration early in training scripts to prevent runtime errors

### Memory Management
- Always consider VRAM limitations when implementing features
- Use gradient checkpointing for large models
- Implement block swapping for models with transformer architectures
- Cache intermediate results (latents, text embeddings) when possible