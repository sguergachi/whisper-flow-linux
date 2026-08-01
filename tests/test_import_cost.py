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
    """Module names loaded after running `statement` in a new interpreter."""
    code = (
        "import sys\n"
        f"{statement}\n"
        "print('\\n'.join(sorted(sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
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
