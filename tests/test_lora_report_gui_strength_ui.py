import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QAbstractItemView, QApplication

import sdxl_lora_report_gui as gui


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_strength_editor_exposes_meaningful_modes(app):
    editor = gui.StrengthEditorWidget()

    editor.set_strength_spec((0.8,))
    assert editor.mode_combo.currentText() == "Common"
    assert editor.strength_spec() == (0.8,)

    editor.set_strength_spec((1.0, 0.5))
    assert editor.mode_combo.currentText() == "TE / U-Net"
    assert editor.strength_spec() == (1.0, 0.5)

    editor.set_strength_spec((1.0, 0.25, 0.5))
    assert editor.mode_combo.currentText() == "TE1 / TE2 / U-Net"
    assert editor.strength_spec() == (1.0, 0.25, 0.5)


def test_switching_mode_carries_resolved_values_forward(app):
    editor = gui.StrengthEditorWidget()
    editor.set_strength_spec((1.0, 0.5))
    editor.mode_combo.setCurrentIndex(2)
    assert editor.strength_spec() == (1.0, 1.0, 0.5)


def test_strength_editor_preserves_large_and_high_precision_values(app):
    editor = gui.StrengthEditorWidget()
    editor.set_strength_spec((20.0, 0.333333333333, -12.5))

    assert editor.strength_spec() == (20.0, 0.333333333333, -12.5)


def test_condition_tree_reserves_double_click_for_strength_dialog(app, tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "GUI_CONFIG_PATH", tmp_path / "last.json")
    monkeypatch.setattr(gui, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(gui, "QUEUE_STATE_PATH", tmp_path / "queue" / "state.json")
    window = gui.MainWindow()
    try:
        triggers = window.condition_tree.editTriggers()
        assert triggers & QAbstractItemView.EditKeyPressed
        assert not triggers & QAbstractItemView.DoubleClicked
    finally:
        window.close()


def test_condition_creation_uses_current_defaults_not_stale_asset_values(app, tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "GUI_CONFIG_PATH", tmp_path / "last.json")
    monkeypatch.setattr(gui, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(gui, "QUEUE_STATE_PATH", tmp_path / "queue" / "state.json")
    window = gui.MainWindow()
    try:
        asset = gui.LoraAsset("asset", "asset", "asset.safetensors", (0.8,), "XLMLT1")
        window.default_strength_editor.set_strength_spec((1.0, 0.25, 0.5))
        window.default_lbw_combo.setCurrentText("XLMIDD")

        item = window.condition_item_from_asset(asset)

        assert item.strength == (1.0, 0.25, 0.5)
        assert item.lbw == "XLMIDD"
    finally:
        window.close()
