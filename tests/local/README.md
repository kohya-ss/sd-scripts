# Local-only regression tests

The tests in this directory need model weights on the local disk and are **not run by CI** (`pytest` from the repository root does not descend into `tests/local`). They exist to check dependency upgrades (transformers, diffusers, torch, ...) against the outputs of the versions currently in use.

## Setup

```bash
cp tests/local/models.example.toml tests/local/models.toml
# edit tests/local/models.toml: fill in the paths of the models you have, delete the rest
```

`models.toml` and the recorded references (`tests/local/references/`) are git-ignored.

The tokenizers are fetched from the Hugging Face Hub on the first run and cached; afterwards `HF_HUB_OFFLINE=1` works.

## Usage

```bash
# 1. record references with the current dependency versions
python tests/local/regression_te.py record            # all entries + schedulers
python tests/local/regression_te.py record --models sdxl,flux

# 2. upgrade the dependencies, then compare
python tests/local/regression_te.py compare
pytest tests/local                                    # same thing as pytest
pytest tests/local -k sdxl

python tests/local/regression_te.py list              # show the configured entries
```

## What is compared

- **Text encoders** (per configured model): the trainers' own `TokenizeStrategy.tokenize` -> `TextEncodingStrategy.encode_tokens` path with the text encoders built by the family's loader, for a fixed prompt set (empty, short, tags, Japanese + emoji, > 77 tokens, > 225 tokens), one prompt at a time and once as a batch. Token ids / attention masks must match exactly; hidden states, pooled outputs etc. are compared with `numpy.allclose` using per-model tolerances.
- **VAE** (sd / sdxl): encode / decode of a fixed image through the diffusers `AutoencoderKL` built by the checkpoint loader (fp32).
- **Schedulers** (always, no weights needed): the diffusers noise schedulers from `library.sampling.get_my_scheduler` for every sampler name, with and without v-prediction: timesteps, `alphas_cumprod`, `sigmas`, and the sample after a few `step()` calls with fixed noise.

For every array the maximum absolute / relative difference and the cosine similarity are printed, so drifts below the tolerance remain visible. The reference `.json` next to each `.npz` records the library versions the reference was made with.

The prompt list in `regression_te.py` must not change once references are recorded (the comparison refuses to run if it did; re-record in that case).
