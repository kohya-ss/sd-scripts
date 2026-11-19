import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import analyze_run_loss  # noqa: E402


def write_log(path, runs):
    with open(path, "w", encoding="utf-8") as f:
        for run_id, rows in runs:
            f.write(json.dumps({"run_id": run_id, "ts": 0.0}) + "\n")
            for row in rows:
                f.write(json.dumps(row) + "\n")


def make_row(image, epoch, timestep, loss, size=(512, 512), batch_size=2):
    return {
        "dataset": 0,
        "subset": 1 if image.startswith("dir_b") else 0,
        "step": epoch,
        "epoch": epoch,
        "image": image,
        "timestep": timestep,
        "loss": loss,
        "width": size[0],
        "height": size[1],
        "batch_size": batch_size,
        "loss_weight": 1.0,
    }


def make_rows():
    """The loss is 0.1 at low timesteps and 1.0 at high timesteps. `hard.png` is always 50% higher, `improving.png`
    starts 60% higher and goes down to the average."""
    rows = []
    for epoch in range(1, 5):
        for timestep, base in [(100.0, 0.1), (900.0, 1.0)]:
            rows.append(make_row("dir_a/normal1.png", epoch, timestep, base))
            rows.append(make_row("dir_a/normal2.png", epoch, timestep, base))
            rows.append(make_row("dir_b/hard.png", epoch, timestep, base * 1.5, size=(768, 512)))
            rows.append(make_row("dir_b/improving.png", epoch, timestep, base * (1.8 - 0.2 * epoch), size=(768, 512)))
    return rows


def test_load_runs_selects_by_run_id(tmp_path):
    path = tmp_path / "loss.jsonl"
    write_log(path, [("run1", [make_row("a.png", 1, 100.0, 0.1)]), ("run2", make_rows())])
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**make_row("nan.png", 1, 100.0, 0.1), "loss": None}) + "\n")  # non-finite loss
        f.write('{"step": 5, "epoch": 1, "image": "trunc')  # incomplete last line of a running training

    runs = analyze_run_loss.load_runs(str(tmp_path))  # the run directory is also accepted
    assert list(runs.keys()) == ["run1", "run2"]
    assert len(runs["run1"]) == 1
    assert len(runs["run2"]) == 32


def test_residual_removes_the_effect_of_the_timestep():
    result = analyze_run_loss.analyze(make_rows(), num_bins=10)

    assert result["summary"]["samples"] == 32
    assert result["summary"]["images"] == 4
    assert [b["count"] for b in result["by_timestep"]] == [16, 16]

    images = {os.path.basename(x["image"]): x for x in result["by_image"]}
    # the raw loss of every image is dominated by the timestep, the residual is not
    assert images["hard.png"]["mean_residual"] > images["improving.png"]["mean_residual"] > images["normal1.png"]["mean_residual"]
    assert images["normal1.png"]["mean_residual"] == pytest.approx(images["normal2.png"]["mean_residual"])
    assert result["by_image"][0]["image"] == "dir_b/hard.png"  # sorted by the residual

    assert images["hard.png"]["slope"] == pytest.approx(0.0, abs=1e-6)  # stays hard
    assert images["improving.png"]["slope"] < -0.1  # is being learned
    assert images["hard.png"]["count"] == 8 and images["hard.png"]["epochs"] == 4


def test_group_tables():
    result = analyze_run_loss.analyze(make_rows())

    by_resolution = {x["resolution"]: x for x in result["by_resolution"]}
    assert set(by_resolution) == {"512x512", "768x512"}
    assert by_resolution["768x512"]["images"] == 2
    assert by_resolution["768x512"]["mean_residual"] > by_resolution["512x512"]["mean_residual"]

    assert [x["directory"] for x in result["by_directory"]] == ["dir_a", "dir_b"]
    assert [(x["subset"], x["images"]) for x in result["by_subset"]] == [("0-0", 2), ("0-1", 2)]
    assert [x["batch_size"] for x in result["by_batch_size"]] == [2]


def test_slope_needs_three_epochs():
    rows = [make_row("a.png", epoch, 100.0, 0.1 * epoch) for epoch in (1, 2)]
    assert analyze_run_loss.analyze(rows)["by_image"][0]["slope"] is None


def test_rows_without_timestep_or_size():
    rows = [{"step": 1, "epoch": 1, "image": "a.png", "timestep": None, "loss": 0.2, "width": None, "height": None}]
    result = analyze_run_loss.analyze(rows)
    assert result["by_image"][0]["mean_residual"] == pytest.approx(0.0)
    assert result["by_resolution"][0]["resolution"] == "(unknown)"
    assert result["by_subset"][0]["subset"] == "(unknown)"


def test_main_prints_report_and_json(tmp_path, monkeypatch, capsys):
    path = tmp_path / "loss.jsonl"
    write_log(path, [("run1", make_rows())])

    monkeypatch.setattr(sys, "argv", ["analyze_run_loss.py", str(path), "--top", "2"])
    analyze_run_loss.main()
    out = capsys.readouterr().out
    assert "run_id: run1" in out
    assert "Images with the highest residual" in out
    assert "hard.png" in out

    monkeypatch.setattr(sys, "argv", ["analyze_run_loss.py", str(path), "--json"])
    analyze_run_loss.main()
    result = json.loads(capsys.readouterr().out)
    assert result["run_id"] == "run1"
    assert len(result["by_image"]) == 4

    monkeypatch.setattr(sys, "argv", ["analyze_run_loss.py", str(path), "--run_id", "missing"])
    with pytest.raises(SystemExit):
        analyze_run_loss.main()
