"""What the frozen Windows build must contain, and how it must call GTK.

PyInstaller finds dependencies by reading the source, so it only bundles
imports it can see. Anything deliberately imported at the point of use to
keep it off the startup path is invisible to that analysis and has to be
declared, or the build is missing it - and the failure surfaces on a user's
machine, in the one code path that needed it.

The GTK checks below are here for the same reason rather than beside the
window tests: those need GTK, libadwaita and a display, so they skip on a
Windows runner and prove nothing about the platform they exist for. These
read the source and always run.
"""

import re
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[1] / "packaging/windows/whisper-flow.spec"

# Third-party modules the code imports lazily, and where it does it.
LAZY_IMPORTS = {
    "pystray": "whisper_flow/daemon.py",
    "velopack": "whisper_flow/updater.py",
}


def _hidden_imports() -> str:
    return SPEC.read_text(encoding="utf-8")


@pytest.mark.parametrize("module,where", sorted(LAZY_IMPORTS.items()))
def test_a_lazily_imported_module_is_declared_to_pyinstaller(module, where):
    assert f'"{module}"' in _hidden_imports(), (
        f"{module} is imported lazily in {where}, so PyInstaller cannot see "
        f"it; add it to hiddenimports or the frozen build will not have it"
    )


def test_the_lazy_list_matches_what_the_code_actually_does():
    """Catches a module being made lazy without this list being updated."""
    src = Path(__file__).resolve().parents[1] / "src/whisper_flow"
    found = set()
    for path in src.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            if not line.startswith((" ", "\t")):
                continue                    # module scope: PyInstaller sees it
            match = re.match(r"(?:from|import)\s+([a-zA-Z_][\w]*)", stripped)
            if match and match.group(1) in LAZY_IMPORTS:
                found.add(match.group(1))
    assert found, "no lazy imports detected; has the mechanism changed?"
    for module in found:
        assert module in LAZY_IMPORTS


# GTK namespaces the Linux build alone uses: the layer-shell protocol has no
# Windows equivalent, and GLibUnix is what its name says. Both sit behind a
# sys.platform guard, so neither belongs in the Windows bundle.
LINUX_ONLY_NAMESPACES = {"Gtk4LayerShell", "GLibUnix"}


def _namespaces_the_ui_needs() -> set[str]:
    """Every gi namespace the source asks for, however it asks."""
    src = Path(__file__).resolve().parents[1] / "src/whisper_flow"
    found = set()
    for path in src.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        found |= set(re.findall(r"require_version\(\s*[\"'](\w+)[\"']", text))
        for imported in re.findall(r"from gi\.repository import ([^\n#]+)", text):
            for name in imported.split(","):
                name = name.split(" as ")[0].strip()
                if name.isidentifier():
                    found.add(name)
    return found - LINUX_ONLY_NAMESPACES


def test_every_gtk_namespace_the_ui_imports_ships_a_typelib():
    """Typelibs are data, so nothing about the build notices one missing.

    Gdk-4.0 was left out of the spec while three modules imported Gdk. The
    build stayed green, the installer built, and the app raised "Namespace
    Gtk not available" at the first window on the user's machine.
    """
    spec = SPEC.read_text(encoding="utf-8")
    needed = _namespaces_the_ui_needs()
    assert "Gtk" in needed and "Gdk" in needed, (
        "namespace detection found neither Gtk nor Gdk; has the import style "
        "changed?"
    )
    missing = [ns for ns in sorted(needed) if f'"{ns}-' not in spec]
    assert not missing, (
        f"the UI imports {', '.join(missing)} but the spec bundles no typelib "
        f"for them; the frozen build will fail at the first window"
    )


def test_the_typelib_path_is_not_left_to_pyinstaller():
    """PyInstaller's gi rthook assigns GI_TYPELIB_PATH before this runs.

    It points at its own gi_typelibs, inferred from an import graph that also
    contains pystray's GTK 3 backend. setdefault() against that is a no-op,
    which is how the curated GTK 4 typelibs went unused in a shipped build.
    """
    entry = (Path(__file__).resolve().parents[1]
             / "src/whisper_flow/__main__win__.py").read_text(encoding="utf-8")
    assert 'setdefault(\n        "GI_TYPELIB_PATH"' not in entry
    assert "setdefault(\"GI_TYPELIB_PATH\"" not in entry
    assert 'os.environ["GI_TYPELIB_PATH"] =' in entry, (
        "the bundled typelib directory must be assigned, and placed ahead of "
        "whatever PyInstaller's runtime hook set"
    )


def test_the_cairo_foreign_struct_bridge_is_declared():
    """The overlay is one cairo draw callback; without this it draws nothing.

    PyGObject converts GTK's cairo_t into a cairo.Context through gi._gi_cairo,
    which its C code imports under a name assembled at runtime. PyInstaller
    sees no such import, left it out, and the shipped overlay raised
    "Couldn't find foreign struct converter for 'cairo.Context'" on every
    frame - a window that was created, styled and blank.
    """
    assert "gi._gi_cairo" in SPEC.read_text(encoding="utf-8"), (
        "gi._gi_cairo is imported by name from PyGObject's C code, so "
        "PyInstaller cannot see it; without it the overlay never paints"
    )


def _window_modules() -> list[Path]:
    src = Path(__file__).resolve().parents[1] / "src/whisper_flow"
    return [src / "hud_app.py", src / "settings_gtk.py"]


def _outside_the_helper(text: str) -> list[tuple[int, str]]:
    """Numbered lines, minus _load_css's body.

    The helper keeps the one-argument call as its own fallback, for the GTK
    versions where it is the correct one. That is the single place allowed to
    name it, so it is cut out before looking for anyone else doing so.
    """
    lines = text.splitlines()
    inside = False
    kept = []
    for number, line in enumerate(lines, 1):
        if line.startswith("def _load_css("):
            inside = True
            continue
        if inside:
            if line and not line[0].isspace():
                inside = False          # back at module scope
            else:
                continue
        kept.append((number, line))
    return kept


def test_css_is_not_loaded_through_the_one_argument_call():
    """load_from_data(css) stopped working, and took both windows with it.

    The length was always a separate argument; older introspection data
    annotated it as the array's own length so PyGObject supplied it. Newer
    data does not, and the call raises TypeError - which killed the overlay
    inside GTK and left the settings window unbuilt inside an activate
    handler, with the process still running and nothing on screen.
    """
    offenders = []
    for path in _window_modules():
        for number, line in _outside_the_helper(
                path.read_text(encoding="utf-8")):
            if re.search(r"\bload_from_data\([^,)]+\)", line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        "load_from_data() with one argument raises TypeError on current "
        "PyGObject; go through the _load_css helper: " + ", ".join(offenders)
    )


def test_every_window_loads_its_css_through_the_helper():
    """A new window pasting the old call back in is the way this returns."""
    for path in _window_modules():
        text = path.read_text(encoding="utf-8")
        assert "def _load_css(" in text, f"{path.name} has no _load_css helper"
        assert text.count("_load_css(provider") >= 1, (
            f"{path.name} defines _load_css but does not use it")


def test_text_is_read_and_written_as_utf8_everywhere():
    """Windows defaults to cp1252, and these files hold arrows and check marks.

    Reading the diagnostics log with the platform default raised
    UnicodeDecodeError on Windows - while reporting a failure, which is the
    one moment it must not add another.
    """
    src = Path(__file__).resolve().parents[1] / "src/whisper_flow"
    offenders = []
    for path in src.glob("*.py"):
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.(read_text|write_text)\(\s*\)", line):
                offenders.append(f"{path.name}:{number}")
            elif re.search(r"\.write_text\([^)]*\)", line) and \
                    "encoding" not in line and "join" in line:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        "text read or written with the platform encoding: " + ", ".join(offenders))
