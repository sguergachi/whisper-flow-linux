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
LINUX_SPEC = (
    Path(__file__).resolve().parents[1] / "packaging/linux/whisper-flow.spec"
)
LINUX_ENTRY = (
    Path(__file__).resolve().parents[1]
    / "src/whisper_flow/__main_linux__.py"
)

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


def test_the_linux_entry_also_assigns_the_typelib_path():
    """Same trap as Windows: setdefault loses to PyInstaller's rthook.

    The Linux build is thin - no typelibs in the bundle - so the fix is the
    reverse of Windows: the rthook's GI_TYPELIB_PATH must be scrubbed (not
    setdefault-assigned) for gi to find the host's typelibs.
    """
    entry = LINUX_ENTRY.read_text(encoding="utf-8")
    assert 'os.environ.pop(key, None)' in entry
    assert '"GI_TYPELIB_PATH"' in entry
    assert 'setdefault("GI_TYPELIB_PATH"' not in entry
    assert 'os.environ["GI_TYPELIB_PATH"] =' not in entry


def test_the_linux_spec_declares_lazy_pystray():
    """Linux freeze is a separate spec; it must hide the same lazy imports."""
    text = LINUX_SPEC.read_text(encoding="utf-8")
    assert '"pystray"' in text
    assert "hotkey_evdev" in text
    assert "hotkey_win" in text  # excluded, not included as a hiddenimport
    assert "whisper_flow.hotkey_win" in text  # appears under EXCLUDES


def _linux_spec_is_host_lib() -> "object":
    """Evaluate the Linux spec's bundling filter without importing the spec.

    The spec is a PyInstaller script, not a module, so importing it would run
    the whole Analysis - and it imports PyInstaller, which the test
    environments do not install. Only the definitions up to the first helper
    are plain Python; executing just those yields the _is_host_lib decision
    function without ever touching PyInstaller.
    """
    lines = LINUX_SPEC.read_text(encoding="utf-8").splitlines(keepends=True)
    code = []
    for line in lines:
        if line.startswith("def gtk_runtime"):
            break
        if line.startswith("from PyInstaller"):
            continue
        code.append(line)
    namespace = {"re": re}
    exec("".join(code), namespace)  # noqa: S102 - static regex definitions
    return namespace["_is_host_lib"]


def test_the_linux_bundle_never_ships_the_openssl_the_host_already_has():
    """A bundled Ubuntu-era libssl shadows the host's and kills host libs.

    On Arch/Fedora/current Ubuntu the host libcurl and libsoup need
    OPENSSL_3.2.0; the AppImage bundled the 22.04 libssl.so.3, the loader
    resolved it first (LD_LIBRARY_PATH is fixed at process start), and every
    GTK typelib load died with "version 'OPENSSL_3.2.0' not found". The whole
    family must be skipped so the host provides it.
    """
    is_host_lib = _linux_spec_is_host_lib()
    for name in (
        "libssl.so.3",
        "libcrypto.so.3",
        "libgnutls.so.30",
        "libz.so.1",
        "libbz2.so.1.0",
        "libexpat.so.1",
        "libffi.so.8",
        "liblzma.so.5",
        "libzstd.so.1",
        "libbrotlicommon.so.1",
        "libstdc++.so.6",
        "libgcc_s.so.1",
        "libasound.so.2",
        "libjack.so.0",
    ):
        assert is_host_lib(name), (
            f"{name} is a system library the host provides on every distro "
            f"this AppImage targets; bundling it shadows the host's copy and "
            f"breaks loaders that need a newer one"
        )


def test_the_linux_bundle_keeps_pyinstallers_hashed_libraries():
    """The skip list names jpeg/png16/tiff/freetype and friends by basename.

    PyInstaller renames collision-prone libs to name-<hash>.so and rewires
    every reference to the hashed name, so those can never shadow the host
    and must survive the filter. When the filter stripped them too, it left
    dangling symlinks behind and Pillow could not import - the daemon died on
    every machine, including ones whose system libraries were fine.
    """
    is_host_lib = _linux_spec_is_host_lib()
    for name in (
        "libtiff-fc87e79d.so.6.2.0",
        "libjpeg-31e2ca52.so.62.4.0",
        "libpng16-abb096d5.so.16.58.0",
        "libfreetype-9fc94c80.so.6.20.6",
        "libharfbuzz-172d1f63.so.0.61421.0",
        "libzstd-44be1190.so.1.5.7",
        "libXau-154567c4.so.6.0.0",
    ):
        assert not is_host_lib(name), (
            f"{name} is a hashed/renamed lib: keeping it is what stops the "
            f"pillow.libs symlinks from dangling and Pillow from failing"
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


def test_the_speech_server_cannot_outlive_the_daemon():
    """stop() only runs on a clean shutdown, and that is the easy case.

    Killed, crashed, or killed by an installer wanting its files back, the
    server was orphaned: still holding the port, still holding a model, still
    taking threads. Two were found running at once, one with five minutes of
    CPU behind it, which is most of what "inference got slow" was.
    """
    backend = (Path(__file__).resolve().parents[1]
               / "src/whisper_flow/backend.py").read_text(encoding="utf-8")
    assert "KILL_ON_JOB_CLOSE" in backend or "_kill_on_exit_job" in backend, (
        "the server must be started inside a job that dies with us; stop() "
        "alone never runs when the daemon is killed")
    assert "_adopt_into_job" in backend, (
        "creating the job is not enough - the server has to be put in it")
    assert "stop_strays" in backend, (
        "a server left by an earlier run still holds the port, and the "
        "readiness check will happily connect to it and report success")
    assert "stop_managed_strays" in backend, (
        "CPU and CUDA engines live in different directories; clearing only the "
        "engine about to start leaves the other one thrashing the same port")
    assert "stop_port_whisper_servers" in backend, (
        "a hand-built whisper-server on our port is invisible to path match "
        "and must still be cleared before we claim the port")
    # Linux used to early-return 0 from stop_strays; the real walk must exist.
    assert "_stop_strays_linux" in backend, (
        "Linux must reap orphan servers the same way Windows does")


def test_a_velopack_hook_exits_instead_of_becoming_the_daemon():
    """The installer waits 30 seconds for the hook process to end.

    velopack.App().run() returns rather than exiting, so control carried on
    into the daemon and the hook never came back. The installer killed it
    and told the user "Install Partially Succeeded" - on an install that had
    already copied every file, made every shortcut and written the uninstall
    key. Nothing was wrong except that the app would not leave.
    """
    entry = (Path(__file__).resolve().parents[1]
             / "src/whisper_flow/__main__win__.py").read_text(encoding="utf-8")
    assert "--veloapp-" in entry, (
        "the entry point must recognise Velopack's hook arguments")
    hook_index = entry.index("--veloapp-")
    daemon_index = entry.index("WhisperFlowDaemon().run(")
    between = entry[hook_index:daemon_index]
    assert "return 0" in between, (
        "a --veloapp-* hook must return before the daemon starts, or the "
        "installer waits 30s, kills it, and reports a partial install")


def test_the_icon_theme_ships_its_index_and_keeps_its_shape():
    """653 icons and no index.theme is not a theme, it is a directory.

    GTK finds icons through index.theme, which names the subdirectories they
    live in. The spec globbed symbolic/**/*.svg into one flat destination and
    shipped no index.theme at all, so every icon the settings window named
    came out as the broken-image glyph - except the handful GTK compiles into
    libgtk, which need no theme and made it look like most of it worked.
    """
    spec = SPEC.read_text(encoding="utf-8")
    assert "index.theme" in spec, (
        "no index.theme is bundled; GTK will not see an icon theme and every "
        "named icon falls back to the broken-image glyph")
    assert "add_tree(" in spec, (
        "the icon tree must be copied with its subdirectories intact; add() "
        "flattens every match into one directory")
    flattened = re.search(r'add\(r"share\\icons\\[^"]*\*\*', spec)
    assert not flattened, (
        f"{flattened.group(0) if flattened else ''} flattens the icon tree "
        f"into a single directory; use add_tree")


def test_the_icons_the_window_names_live_in_one_list():
    """So the frozen build can check them, rather than a copy that drifts."""
    settings = (Path(__file__).resolve().parents[1]
                / "src/whisper_flow/settings_gtk.py").read_text(encoding="utf-8")
    entry = (Path(__file__).resolve().parents[1]
             / "src/whisper_flow/__main__win__.py").read_text(encoding="utf-8")
    assert "ICON_NAMES" in settings and "SECTION_ICONS" in settings, (
        "settings_gtk must name its icons in one module-level list")
    assert "ICON_NAMES" in entry, (
        "the selftest must check the icons against that list, or a missing "
        "icon ships and is only found by looking at the window")


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


def test_apprun_only_preloads_the_layer_shell_for_gtk4_roles():
    """The daemon must never load gtk4-layer-shell - it drags GTK4 in.

    libgtk4-layer-shell links GTK4, and once GTK4 is in the daemon's
    process pystray's tray backend (GTK3) resolves its gdk_display_manager_
    get call to GTK4's implementation, which aborts with
    "gdk_display_manager_get() was called before gtk_init()" - the whole
    daemon, tray and hotkeys, dies on the user's desktop. Only the GTK4
    windows (--hud, --settings) need the preload, and only they may set it.
    """
    script = (Path(__file__).resolve().parents[1]
              / "packaging/linux/make-appimage.sh").read_text(encoding="utf-8")
    apprun = script.split('<<\'APPRUN\'', 1)[1].split('APPRUN', 1)[0]
    assert 'case " $* " in' in apprun, (
        "AppRun must gate the layer-shell preload on the invocation: the "
        "daemon (no flags) must not load gtk4-layer-shell or GTK4's gdk "
        "interposes pystray's GTK3 call and the daemon aborts"
    )
    assert '" --hud "*|*" --settings "*' in apprun, (
        "the preload must be limited to the GTK4 windows, the HUD and the "
        "settings window"
    )
    assert 'export LD_PRELOAD' in apprun
    # The export must live inside the case, never before it.
    before_case = apprun.split('case " $* " in', 1)[0]
    assert "export LD_PRELOAD" not in before_case, (
        "an unconditional preload poisons the daemon: GTK4 loads at process "
        "start, and pystray's GTK3 tray backend then aborts on GTK4's "
        "gdk_display_manager_get"
    )
