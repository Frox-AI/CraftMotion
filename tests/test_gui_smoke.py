"""Headless GUI smoke test using a stub tkinter (no display in CI)."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mocktk


@pytest.fixture(scope="module")
def app():
    mocktk.install()
    from craftmotion.gui.app import MainWindow
    window = MainWindow()
    mocktk.QUEUE.clear()
    yield window


def test_window_builds(app):
    assert app.reg is not None
    assert app.timeline is not None and app.preview is not None and app.inspector is not None


def test_edit_flow(app):
    app.refresh_all()
    app.set_time(300)
    app.add_layer("text")
    app.on_project_changed()
    assert len(app.reg.project.layers) >= 1
    app.inspector.rebuild()
    app.inspector.refresh_values()
    app.timeline.redraw()
    app.timeline.zoom(1.3)
    app.timeline.fit()
    app.media.refresh()
    app.undo()
    app.redo()


def test_preview_and_agent(app):
    app.preview.request_render(force=True)
    w, h = app.preview.view_size()
    assert w > 0 and h > 0
    cfg = app.agent._stdio_config()
    assert "mcpServers" in cfg
    info = app.agent.bridge.start()
    assert info["running"]
    app.agent.bridge.stop()


def test_storyboard_command(app):
    res = app.run_command("render.storyboard", {"count": 2, "cols": 2, "tile": 80,
                                                "path": os.path.join(tempfile.gettempdir(),
                                                                     "gui_smoke_board.png")})
    assert os.path.exists(res["path"])
