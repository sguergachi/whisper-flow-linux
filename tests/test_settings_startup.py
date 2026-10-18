"""Exercise GApplication registration on a private bus and test display."""

import os
import shutil
import subprocess
import sys

import pytest

_CHILD = '''
import os
from pathlib import Path
from whisper_flow.__main_linux__ import _bootstrap_gtk_runtime
_bootstrap_gtk_runtime()
from whisper_flow import settings_gtk as m
def event(name):
    with open(os.environ["TEST_EVENTS"], "a") as f:
        f.write(name + "\\n")
class Window(m.Gtk.ApplicationWindow):
    _show_t0 = 0
    def __init__(self, application, config):
        super().__init__(application=application)
        event("built")
    def realize(self):
        super().realize()
        event("ready")
    def show_for_click(self, *args):
        event("show")
        self.present()
        return False
m.SettingsWindow = Window
raise SystemExit(m.main())
'''


def test_duplicate_prewarm_never_activates_existing_settings(tmp_path):
    if not os.environ.get("DISPLAY") or not shutil.which("dbus-run-session"):
        pytest.skip("needs an isolated test display and dbus-run-session")
    pytest.importorskip("gi")
    child = tmp_path / "settings_child.py"
    child.write_text(_CHILD)
    driver = tmp_path / "driver.py"
    # The entire interaction runs on a private bus, so this test can never
    # activate a user's real Settings window through its well-known bus name.
    driver.write_text('''
import os, subprocess, sys, time
from pathlib import Path
env = os.environ.copy()
env["WHISPER_FLOW_SETTINGS_RESIDENT"] = "1"
events = Path(env["TEST_EVENTS"])
def wait_for(name):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if events.exists() and name in events.read_text().splitlines():
            return
        time.sleep(.02)
    raise AssertionError("missing " + name)
with open(events.with_suffix(".log"), "w") as log:
    primary = subprocess.Popen([sys.executable, sys.argv[1]], env=env,
        stdin=subprocess.PIPE, stdout=log, stderr=log, text=True)
    try:
        wait_for("ready")
        duplicate = subprocess.run([sys.executable, sys.argv[1]], env=env,
            stdout=log, stderr=log, text=True, timeout=20)
        assert "show" not in events.read_text().splitlines()
        assert events.read_text().splitlines().count("built") == 1
        assert primary.poll() is None
        # An explicit standalone launch must still raise the same window.
        env.pop("WHISPER_FLOW_SETTINGS_RESIDENT")
        requested = subprocess.run([sys.executable, sys.argv[1]], env=env,
            stdout=log, stderr=log, text=True, timeout=20)
        assert requested.returncode == 0
        wait_for("show")
        assert events.read_text().splitlines().count("built") == 1
        primary.stdin.write("quit\\n")
        primary.stdin.flush()
        primary.wait(timeout=5)
    finally:
        if primary.poll() is None:
            primary.kill()
            primary.wait(timeout=5)
''')
    env = os.environ.copy()
    env["TEST_EVENTS"] = str(tmp_path / "events.txt")
    env["WHISPER_FLOW_CONFIG_DIR"] = str(tmp_path / "config")
    env["GDK_BACKEND"] = "x11"
    env["GSK_RENDERER"] = "cairo"
    env["NO_AT_BRIDGE"] = "1"
    env["GTK_A11Y"] = "none"
    driver_log = tmp_path / "driver.log"
    with driver_log.open("w") as output:
        result = subprocess.run(
            ["dbus-run-session", "--", sys.executable, str(driver), str(child)],
            env=env, stdout=output, stderr=output, text=True, timeout=65, check=False)
    log_path = (tmp_path / "events.txt").with_suffix(".log")
    assert result.returncode == 0, driver_log.read_text() + (
        log_path.read_text() if log_path.exists() else "")
