from __future__ import annotations

import json
from pathlib import Path

from tools.render_dq_profile_anonymized_example import render


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY_ROOT / "docs" / "examples" / "dq_dataset_profiler_anonymized_data.json"


def test_anonymized_example_is_self_contained_and_explains_uncertainty() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rendered = render(payload)
    assert len(payload["datasets"]) == 7
    assert rendered.count('<article class="dataset-card">') == 7
    assert rendered.count('<svg class="chart"') == 7
    assert "source-cluster bootstrap 95%区間" in rendered
    assert "量子化が必ず悪いという意味ではありません" in rendered
    assert "画質比較ではありません" in rendered
    assert "同一画像・タグ設計だけを変えたpaired例" in rendered
    assert "距離1.0（画質の合否線ではない）" in rendered
    assert "固定Y軸0–4.0" in rendered

    # The example must stay portable and must not expose local provenance.
    assert "C:\\" not in rendered
    assert "D:\\" not in rendered
    assert "file://" not in rendered
    assert "https://" not in rendered
    assert "<script" not in rendered
    assert "<img" not in rendered
    assert "dataset_config" not in rendered


def test_anonymized_fixture_has_only_generic_dataset_ids_and_rounded_values() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["anonymization"]["rounded_values"] is True
    for dataset in payload["datasets"]:
        assert str(dataset["id"]).startswith("Dataset ")
        for candidate in dataset["candidates"]:
            for key in ("body", "tail"):
                assert round(float(candidate[key]), 3) == float(candidate[key])
            for key in ("body_ci", "tail_ci"):
                assert len(candidate[key]) == 2
                assert all(round(float(value), 3) == float(value) for value in candidate[key])
