"""pytest wrapper for the local regression harness (see regression_te.py).

Skipped entirely unless ``tests/local/models.toml`` exists. Each model listed
there (plus the weight-free ``schedulers`` check) becomes one test that compares
the current outputs against the recorded references. Tests whose reference has
not been recorded yet are skipped.

    pytest tests/local            # run all
    pytest tests/local -k sdxl    # one model
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regression_te as rt  # noqa: E402

if not rt.DEFAULT_CONFIG.exists():
    pytest.skip(f"{rt.DEFAULT_CONFIG} not found (local-only regression tests)", allow_module_level=True)

_CONFIG = rt.load_config()
_SPECS = rt.all_specs(_CONFIG)
_REF_DIR = rt.reference_dir(_CONFIG)


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_regression(spec: rt.ModelSpec):
    if not (_REF_DIR / f"{spec.name}.npz").exists():
        pytest.skip(f"no reference recorded for {spec.name}; run `python tests/local/regression_te.py record --models {spec.name}`")
    result = rt.compare_spec(spec, _REF_DIR)
    print(result.summary)
    assert result.ok, "\n".join(result.failures)
