"""The overlay process must not pay for the whole application.

The HUD is launched on the path that starts a recording and is the thing the
user waits to see. It needs tkinter (Windows) or GTK (Linux) and nothing
else, but importing the package used to drag in the openai SDK, pyaudio,
pystray and PIL first - about 740ms here, considerably more in a frozen
build on Windows. These pin the package's import surface so it cannot
quietly go back.
"""

import subprocess
import sys


def _import_in_fresh_process(statement: str) -> set[str]:
    """Module names loaded after running `statement` in a new interpreter.

    Surfaces the child's stderr on failure. It used to use check=True, which
    raised CalledProcessError and threw the traceback away - so a CI failure
    said only "exit status 1" and the reason had to be guessed at.
    """
    code = (
        "import sys\n"
        f"{statement}\n"
        "print('\\n'.join(sorted(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=False)
    assert out.returncode == 0, (
        f"`{statement}` failed in a fresh interpreter "
        f"(exit {out.returncode}):\n{out.stderr.strip()}"
    )
    return set(out.stdout.split())


EXPENSIVE = ("openai", "pyaudio", "pystray", "PIL")


def test_importing_the_package_does_not_drag_in_the_world():
    loaded = _import_in_fresh_process("import whisper_flow")
    assert not [m for m in EXPENSIVE if m in loaded], (
        "whisper_flow imports something heavy at module scope again; the "
        "overlay pays this on every recording"
    )


def test_the_public_names_still_resolve():
    """Laziness must not change the package's interface."""
    import whisper_flow

    for name in whisper_flow.__all__:
        assert getattr(whisper_flow, name) is not None
    assert "Config" in dir(whisper_flow)


def test_an_unknown_name_still_raises_attribute_error():
    import whisper_flow

    try:
        whisper_flow.NoSuchThing
    except AttributeError:
        return
    raise AssertionError("a missing attribute must still raise AttributeError")


def test_the_version_is_available_without_importing_anything():
    loaded = _import_in_fresh_process(
        "import whisper_flow; assert whisper_flow.__version__")
    assert not [m for m in EXPENSIVE if m in loaded]


def test_all_matches_what_is_actually_exported():
    """A hand-written __all__ must not drift from the lazy import table."""
    import whisper_flow

    assert sorted(whisper_flow.__all__) == sorted(whisper_flow._EXPORTS)


def test_the_daemon_does_not_import_the_openai_sdk():
    """Dictation through a local server never uses it, and it cost ~540ms.

    That half second is paid before the tray icon appears and before any
    hotkey is registered, on every launch, by every user - including the
    majority who never touch the hosted API.
    """
    loaded = _import_in_fresh_process("import whisper_flow.daemon")
    assert "openai" not in loaded


def test_the_openai_client_is_still_reachable_when_wanted():
    """Lazy must not mean gone."""
    from whisper_flow import completion

    assert callable(completion._openai_client)


def test_the_daemon_imports_without_a_display():
    """pystray resolves its backend during import and raises with no display.

    So `import whisper_flow.daemon` needed a screen, which broke it on any
    headless machine and in CI - and cost ~126ms that a run never reaching
    the tray does not need.
    """
    import os

    code = "import whisper_flow.daemon; print('ok')"
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=False)
    assert out.returncode == 0, out.stderr.strip()


def test_the_tray_library_is_not_loaded_just_by_importing():
    loaded = _import_in_fresh_process("import whisper_flow.daemon")
    assert "pystray" not in loaded
