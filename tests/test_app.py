from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_shell_renders_without_exception():
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    assert not app.exception
    assert len(app.title) >= 1


def test_attack_comparison_form_runs_and_exports(synthetic_framingham, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data" / "framingham.csv"
    dataset.parent.mkdir()
    synthetic_framingham.to_csv(dataset, index=False)

    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    next(item for item in app.number_input if item.label == "Clients").set_value(5)
    next(item for item in app.number_input if item.label == "Communication rounds").set_value(2)
    next(button for button in app.button if button.label == "Run comparison").click().run()

    assert not app.exception
    assert not app.error
    metrics, _ = app.session_state["attack_comparison_result"]
    assert set(metrics["setting"]) == {"clean", "attack", "defense"}
    assert set(metrics["split"]) == {"validation"}
    assert len(app.get("download_button")) == 1
